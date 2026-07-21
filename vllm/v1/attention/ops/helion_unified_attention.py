# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import helion
import helion.experimental  # noqa: F401  # for aot_kernel (see below)
import helion.language as hl
import torch

from vllm.utils.math_utils import next_power_of_2

from .triton_unified_attention import (
    unified_attention as triton_baseline_unified_attention,
)

# Number of buckets used to discretize the two batch-shape characteristics
# that steer JIT specialization / re-autotuning. Each characteristic is mapped
# to an integer in [0, MIX_NUM_BUCKETS]; more buckets = finer specialization
# (more configs to tune) vs. fewer buckets = coarser (more cache reuse).
MIX_NUM_BUCKETS = 8


def _compute_mix_buckets(
    num_total_query_tokens: int,
    num_decode_tokens: int,
    num_seqs: int,
    max_query_len: int,
) -> tuple[int, int]:
    """Derive two orthogonal batch-shape buckets for kernel specialization.

    Both buckets are host-side integer computations (no GPU sync) so they are
    cheap to evaluate on the hot path.

    Returns:
        A tuple ``(decode_frac_bucket, prefill_skew_bucket)``:

        - ``decode_frac_bucket`` in ``[0, MIX_NUM_BUCKETS]`` captures the
          decode/prefill token ratio: ``0`` = pure prefill, ``MIX_NUM_BUCKETS``
          = pure decode. Unlike the previous ``mix_ratio`` it is independent of
          absolute batch size, so it does not collapse across batch sizes.
        - ``prefill_skew_bucket`` in ``[0, MIX_NUM_BUCKETS]`` captures the shape
          of the prefill query-length distribution as
          ``mean_prefill_qlen / max_prefill_qlen``. Uniform-length prefills map
          to ``MIX_NUM_BUCKETS``; skewed (mixed short/long) prefills map lower.
          It is ``0`` when there are no prefill sequences.
    """
    if num_total_query_tokens <= 0:
        return 0, 0

    decode_frac_bucket = round(
        MIX_NUM_BUCKETS * num_decode_tokens / num_total_query_tokens
    )

    # Each decode sequence contributes exactly one query token, so the number of
    # decode sequences equals num_decode_tokens.
    num_prefill_seqs = num_seqs - num_decode_tokens
    prefill_tokens = num_total_query_tokens - num_decode_tokens
    if num_prefill_seqs > 0 and max_query_len > 0:
        mean_prefill_qlen = prefill_tokens / num_prefill_seqs
        prefill_skew_bucket = round(
            MIX_NUM_BUCKETS * mean_prefill_qlen / max_query_len
        )
        # Clamp to guard against pathological inputs (e.g. a stray decode token
        # counted as prefill making the mean exceed max_query_len).
        prefill_skew_bucket = max(0, min(MIX_NUM_BUCKETS, prefill_skew_bucket))
    else:
        prefill_skew_bucket = 0

    return decode_frac_bucket, prefill_skew_bucket


def _triton_baseline_fn(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    t_query_start_lens,  # [num_seqs+1]
    max_query_len,
    num_seqs,
    q_block_padded_size,
    batch_size_padded,
    decode_frac_bucket,
    prefill_skew_bucket,
):
    max_seqlen = t_seq_lens.max()
    return triton_baseline_unified_attention(
        q=t_query,
        k=t_key_cache,
        v=t_value_cache,
        out=t_output,
        cu_seqlens_q=t_query_start_lens,
        max_seqlen_q=max_query_len,
        seqused_k=t_seq_lens,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=t_block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )



@helion.kernel(
    allow_warp_specialize=True,
    static_shapes=False,
    index_dtype=torch.int64,
    # configs=configs, work with CACHE_DIR instead!
    autotune_baseline_fn=_triton_baseline_fn,
    # autotune_effort="quick",
    # autotune_initial_population_strategy="from_random",
    autotune_effort="full",
    autotune_accuracy_check=False, # still to strict?
    autotune_ignore_errors=True,
    print_repro=False,
    print_output_code=False,
    # autotune_log="",
)
# AOT alternative: use a pretuned heuristic instead of autotuning. Generate it
# with helpers/helion_cache_to_aot.py into helion_heuristics/, then point helion
# at it via HELION_HEURISTIC_DIR=<repo>/helion_heuristics (or copy the file next
# to this source) and run with HELION_AOT_MODE=evaluate. The file must be named
# _helion_aot_helion_unified_attention_<device>_<compute>.py to match
# get_hardware_info() on the target. Do NOT pass key= so the generated
# key_kernel_helion_v9_attention receives the raw kernel args. Swap with the
# @helion.kernel(...) decorator above.
# @helion.experimental.aot_kernel(
#     allow_warp_specialize=True,
#     static_shapes=False,
#     index_dtype=torch.int64,
#     print_repro=False,
#     print_output_code=False,
# )
def kernel_helion_v9_attention(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    t_query_start_lens,  # [num_seqs+1]
    max_query_len,  # must be on CPU
    num_seqs,  # must be on cpu
    # to trigger re-compilation (and re-tuning) for decodes only
    q_block_padded_size: hl.constexpr,
    # to trigger re-compilation (and re-tuning) for small and large batches
    batch_size_padded: hl.constexpr,
    # decode/prefill token-ratio bucket: triggers re-tuning for decode-heavy
    # vs prefill-heavy batches (0 = pure prefill .. MIX_NUM_BUCKETS = pure decode)
    decode_frac_bucket: hl.constexpr,
    # prefill query-length skew bucket: triggers re-tuning when the prefill
    # length distribution changes (uniform vs mixed short/long prompts)
    prefill_skew_bucket: hl.constexpr,
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    q_block_size = hl.register_block_size(1, q_block_padded_size)
    num_pages_at_once = hl.register_block_size(1, 32)

    for seq_tile, tile_m, tile_q in hl.tile(
        [num_seqs, num_query_heads, max_query_len],
        block_size=[1, num_queries_per_kv, q_block_size],
    ):
        seq_idx = seq_tile.begin  # is scalar
        seq_len = t_seq_lens[seq_idx]
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len

        if query_start + tile_q.begin < query_end:
            block_m_size = num_queries_per_kv * q_block_size
            kv_head_idx = tile_m.begin // num_queries_per_kv

            # cannot use tile_q.index directly, since tile_q.index is dynamic
            adjusted_tile_q_index = query_start + tile_q.begin + hl.arange(q_block_size)
            query_head_offset = tile_m.begin + hl.arange(num_queries_per_kv)
            q_load_mask = adjusted_tile_q_index[:, None, None] < query_end
            # (tile_q, tile_m, HEAD_SIZE)
            q = hl.load(
                t_query,
                [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
                extra_mask=q_load_mask,
            )
            # (tile_m, HEAD_SIZE)
            q = q.flatten(start_dim=0, end_dim=1)

            M = hl.full([block_m_size], float("-inf"), dtype=torch.float32)
            L = hl.full([block_m_size], 1.0, dtype=torch.float32)
            acc = hl.zeros([block_m_size, head_size], dtype=torch.float32)

            # adjust for causal mask
            max_seq_prefix_len = context_len + tile_q.begin + block_m_size + 1
            max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
            num_blocks = torch.ceil(max_seq_prefix_len / page_size)
            for tile_n in hl.tile(num_blocks, block_size=num_pages_at_once):
                block_n_size = num_pages_at_once * page_size
                # explicit load due to wrong if tile_n is partial
                blk_idxs = hl.load(
                    t_block_tables,
                    [seq_idx, tile_n.begin + hl.arange(num_pages_at_once)],
                )
                blk_idxs = blk_idxs.view([num_pages_at_once]).to(torch.int64)

                # (tile_n, PAGE_SIZE, 1, HEAD_SIZE)
                k_load = t_key_cache[blk_idxs, :, kv_head_idx, :]
                k_load = k_load.flatten(start_dim=0, end_dim=1)
                # (tile_n, HEAD_SIZE)
                k = hl.zeros([block_n_size, head_size], dtype=k_load.dtype)
                absolute_tile_token_offsets = tile_n.begin * page_size + hl.arange(
                    block_n_size
                )
                k = torch.where(
                    absolute_tile_token_offsets[:, None] < seq_len, k_load, k
                )
                # (HEAD_SIZE, tile_n)
                k = k.transpose(0, 1)

                # (tile_n, PAGE_SIZE, HEAD_SIZE)
                v_load = t_value_cache[blk_idxs, :, kv_head_idx, :]
                v_load = v_load.flatten(start_dim=0, end_dim=1)
                # (tile_n, HEAD_SIZE)
                v = hl.zeros([block_n_size, head_size], dtype=v_load.dtype)
                v = torch.where(
                    absolute_tile_token_offsets[:, None] < seq_len, v_load, v
                )

                # (tile_m, tile_n)
                # use S with float32 as acc to enforce higher precision
                #  for the additions of the dot operation?
                S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
                S = hl.dot(q, k, out_dtype=torch.float32, acc=S) * scale
                block_m_query_mask = tile_q.begin + hl.arange(
                    q_block_size
                ).repeat_interleave(num_queries_per_kv, dim=0)
                # construct 2d causal mask
                causal_mask = (
                    absolute_tile_token_offsets[None, :]
                    < context_len + block_m_query_mask[:, None] + 1
                )
                S = torch.where(causal_mask, S, float("-inf"))

                # (tile_m)
                M_j = torch.maximum(M, torch.amax(S, 1))
                # (tile_m, tile_n)
                P = torch.exp(S - M_j[:, None])
                # (tile_m, )
                L_j = torch.sum(P, 1)
                # (tile_m, )
                alpha = torch.exp(M - M_j)
                # (tile_m, HEAD_SIZE)
                acc = acc * alpha[:, None]
                L = (L * alpha) + L_j
                M = M_j

                # (tile_m, HEAD_SIZE)
                acc = hl.dot(P.to(v.dtype), v, out_dtype=torch.float32, acc=acc)

            # epilogue
            acc = acc / L[:, None]
            hl.store(
                t_output,
                [adjusted_tile_q_index, tile_m.index, hl.arange(head_size)],
                acc.view([q_block_size, num_queries_per_kv, head_size]),
                extra_mask=q_load_mask,
            )


def helion_unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    num_seqs: int,
    num_decode_tokens: int,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    capture_num_seqs: int | None = None,
    capture_max_query_len: int | None = None,
    capture_num_decode_tokens: int | None = None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    assert alibi_slopes is None, "not supported right now, still experimental"
    assert softcap == 0, "not supported right now, still experimental"
    assert k_descale is None, "not supported right now, still experimental"
    assert v_descale is None, "not supported right now, still experimental"
    assert window_size == (-1, -1), "not supported right now, still experimental"

    block_size = v.shape[1]
    assert q.element_size() >= 2 or block_size >= 32, (
        "Block size must be at least 32 for fp8"
    )

    if capture_num_seqs is not None:
        # CUDA graph capture/replay: the batch is already padded to a fixed
        # bucket, so num_seqs and max_query_len should be the values used
        # by the helion compiler. capture_num_decode_tokens carries the TRUE
        # decode-sequence count of the synthetic capture batch (set by the
        # backend), so the mix buckets distinguish a pure-decode capture from a
        # prefill/mixed capture and compile/tune them as separate
        # specializations. Note: the buckets are frozen at capture time; a
        # replayed graph cannot re-specialize if the runtime batch differs.
        batch_size_padded = capture_num_seqs
        max_used_querylen_padded = capture_max_query_len
        mix_num_seqs = capture_num_seqs
        mix_num_decode_tokens = capture_num_decode_tokens
        mix_max_query_len = capture_max_query_len
    else:
        # Eager (non-captured) runs: derive padding from the runtime shapes.
        # trade-off: number of buckets (re-compilation time / JIT jitter) vs.
        # performance.
        max_used_querylen_padded = (
            next_power_of_2(max_seqlen_q)
            if next_power_of_2(max_seqlen_q) in [1, 8, 16, 32, 64]
            else 128
        )

        batch_size_padded_coarse = min(256, next_power_of_2(num_seqs))
        batch_size_padded_fine = (
            min(256, next_power_of_2(num_seqs)) if num_seqs >= 16 else num_seqs
        )
        batch_size_padded = (
            batch_size_padded_coarse if torch.version.cuda else batch_size_padded_fine
        )
        mix_num_seqs = num_seqs
        mix_num_decode_tokens = num_decode_tokens
        mix_max_query_len = max_seqlen_q

    # Two orthogonal constexpr buckets steer JIT specialization / re-autotuning:
    #  - decode_frac_bucket: decode-heavy vs prefill-heavy batches
    #  - prefill_skew_bucket: uniform vs mixed-length prefill distributions
    # q.shape[0] is the total number of query tokens (no GPU sync).
    decode_frac_bucket, prefill_skew_bucket = _compute_mix_buckets(
        num_total_query_tokens=q.shape[0],
        num_decode_tokens=mix_num_decode_tokens,
        num_seqs=mix_num_seqs,
        max_query_len=mix_max_query_len,
    )

    kernel_helion_v9_attention(
        t_output=out,
        t_query=q,
        t_key_cache=k,
        t_value_cache=v,
        t_block_tables=block_table,
        t_seq_lens=seqused_k,
        scale=softmax_scale,
        t_query_start_lens=cu_seqlens_q,
        max_query_len=max_seqlen_q,
        num_seqs=num_seqs,
        q_block_padded_size=max_used_querylen_padded,
        batch_size_padded=batch_size_padded,
        decode_frac_bucket=decode_frac_bucket,
        prefill_skew_bucket=prefill_skew_bucket,
    )

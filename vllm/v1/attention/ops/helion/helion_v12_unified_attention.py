# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""helion_v12: unified paged attention on a *flat q-block grid*.

Difference to v9 (``helion_unified_attention.py``): v9 launches a dense
``[num_seqs, num_query_heads, max_query_len]`` grid, so a skewed mixed batch
(one long prefill + many decodes) launches ``num_seqs * ceil(max_qlen/BLOCK_Q)``
programs of which only ``~sum_i ceil(qlen_i/BLOCK_Q)`` do any work. v12 instead
launches the Triton-style flat q-block grid

    num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs     (host-side upper bound)

and recovers ``(seq_idx, q_block_local_idx)`` from the program id inside the
kernel. See ``kernels/triton_unified_attention.py:929-938`` for the same bound
and ``kernels/triton_attention_helpers.py:60-111`` for the binary-search
inverse that we replace with a vectorized reduction (the batch is bounded by
``MAX_NUM_SEQS``, so a masked load + one sum beats a device-side ``while``).

The bound is deliberately computed on the host from ``q.shape[0]``,
``num_seqs`` and the config's ``BLOCK_Q`` — all frozen at CUDA-graph capture.
A device-computed exact count would make the loop bound data-dependent, and
helion then disallows the ``flat``/``xyz`` pid types to stay cudagraphable
(``helion/language/loops.py:420-424``), which would remove the pid type every
tuned config in ``caches/`` actually uses.
"""

import helion
import helion.language as hl
import torch

from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.ops.triton_unified_attention import (
    unified_attention as triton_baseline_unified_attention,
)

from .helion_unified_attention import (
    MIX_NUM_BUCKETS,  # noqa: F401  # re-exported for schedulers
    _compute_mix_buckets,
    compute_padding,
)

# fmt: off
# ruff-format is disabled for the rest of this file on purpose: helion's
# autotune cache key hashes inspect.getsource() of the kernel function
# (helion/autotuner/local_cache.py), so reflowing a kernel body silently
# invalidates every tuned config for it. Keeping the source byte-identical to
# the research repo (helion-aot-research/kernels/) also keeps the two diffable.

# Upper bound on the batch size the vectorized inverse mapping can resolve.
# The gid -> seq_idx reduction materializes a `num_seqs_padded`-wide vector per
# program, so the bound is what keeps that cost constant instead of a device
# binary search. vLLM's own default max_num_seqs is well below this.
MAX_NUM_SEQS = 256


def _triton_baseline_fn_v12(
    t_output,
    t_query,
    t_key_cache,
    t_value_cache,
    t_block_tables,
    block_table_stride,
    t_seq_lens,
    scale,
    t_query_start_lens,
    num_seqs,
    q_block_padded_size,
    batch_size_padded,
    decode_frac_bucket,
    prefill_skew_bucket,
    num_seqs_padded,
):
    """Autotune baseline: the Triton 2D kernel, as for v9.

    v12 passes the block table flattened and dropped ``max_query_len`` from the
    kernel signature, so both are reconstructed here. The ``max_seqlen_q``
    recovery costs a device->host sync, which is fine: this runs only inside the
    autotuner, never on the hot path.
    """
    max_seqlen = t_seq_lens.max()
    max_seqlen_q = (t_query_start_lens[1:] - t_query_start_lens[:-1]).max()
    block_table = t_block_tables.view(-1, block_table_stride)
    return triton_baseline_unified_attention(
        q=t_query,
        k=t_key_cache,
        v=t_value_cache,
        out=t_output,
        cu_seqlens_q=t_query_start_lens,
        max_seqlen_q=max_seqlen_q,
        seqused_k=t_seq_lens,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )


def compute_autotune_key(
    num_total_query_tokens: int,
    num_decode_tokens: int,
    num_seqs: int,
    max_query_len: int,
    capture_num_seqs: int | None = None,
    capture_max_query_len: int | None = None,
    capture_num_decode_tokens: int | None = None,
) -> tuple[int, int, int, int, int]:
    """v12 specialization key: v9's four buckets plus ``num_seqs_padded``.

    ``num_seqs_padded`` is a function of the ``num_seqs`` that drives the
    launch, and every existing branch already keys at equal-or-finer
    granularity on it (eager CUDA: ``min(256, next_power_of_2(num_seqs))``;
    capture: the exact ``capture_num_seqs``), so it never splits an existing
    bucket on those paths.
    """
    if capture_num_seqs is not None:
        batch_size_padded = capture_num_seqs
        q_block_padded_size = capture_max_query_len
        mix_num_seqs = capture_num_seqs
        mix_num_decode_tokens = capture_num_decode_tokens
        mix_max_query_len = capture_max_query_len
    else:
        q_block_padded_size, batch_size_padded = compute_padding(
            max_query_len, num_seqs
        )
        mix_num_seqs = num_seqs
        mix_num_decode_tokens = num_decode_tokens
        mix_max_query_len = max_query_len

    decode_frac_bucket, prefill_skew_bucket = _compute_mix_buckets(
        num_total_query_tokens=num_total_query_tokens,
        num_decode_tokens=mix_num_decode_tokens,
        num_seqs=mix_num_seqs,
        max_query_len=mix_max_query_len,
    )
    return (
        q_block_padded_size,
        batch_size_padded,
        decode_frac_bucket,
        prefill_skew_bucket,
        next_power_of_2(mix_num_seqs),
    )


@helion.kernel(
    allow_warp_specialize=True,
    static_shapes=False,
    index_dtype=torch.int64,
    autotune_baseline_fn=_triton_baseline_fn_v12,
    autotune_effort="full",
    autotune_accuracy_check=False,
    autotune_ignore_errors=True,
    print_repro=False,
    print_output_code=False,
)
def kernel_helion_v12_attention(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs * max_num_blocks_per_seq] (flattened, see below)
    block_table_stride,  # row stride of the original 2-D block table
    t_seq_lens,  # [num_seqs]
    scale,
    t_query_start_lens,  # [num_seqs+1]
    num_seqs,  # must be on cpu
    # to trigger re-compilation (and re-tuning) for decodes only
    q_block_padded_size: hl.constexpr,
    # to trigger re-compilation (and re-tuning) for small and large batches
    batch_size_padded: hl.constexpr,
    # decode/prefill token-ratio bucket
    decode_frac_bucket: hl.constexpr,
    # prefill query-length skew bucket
    prefill_skew_bucket: hl.constexpr,
    # width of the gid -> seq_idx reduction vector (power of two, <= MAX_NUM_SEQS)
    num_seqs_padded: hl.constexpr,
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

    # Host-side upper bound; same algebra as triton's total_num_q_blocks.
    num_q_blocks = t_query.size(0) // q_block_size + num_seqs

    for q_blk_tile, tile_m in hl.tile(
        [num_q_blocks, num_query_heads],
        block_size=[1, num_queries_per_kv],
    ):
        gid = q_blk_tile.begin  # is scalar

        # Vectorized equivalent of find_seq_idx(..., use_q_block_mode=True):
        # starts[] is monotone in s_idx, so the rank of gid is a masked sum.
        s_idx = hl.arange(num_seqs_padded)
        valid = s_idx < num_seqs
        cu = hl.load(t_query_start_lens, [s_idx], extra_mask=valid)
        starts = cu // q_block_size + s_idx
        seq_idx = torch.sum(((starts <= gid) & valid).to(torch.int32)) - 1

        seq_len = t_seq_lens[seq_idx]
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len
        q_block_offset = (gid - (query_start // q_block_size + seq_idx)) * q_block_size
        # Flat row offset into the block table. The 2-D form
        # `t_block_tables[seq_idx, <vector>]` cannot be lowered when seq_idx is
        # a *device-computed* scalar (helion mixes it with the vector indexer and
        # trips `handle_broadcast_tensor` in
        # helion/_compiler/indexing_strategy.py:1534) — v9 is unaffected because
        # there seq_idx comes from `tile.begin`. Same form as triton's
        # `block_table_offset = seq_idx * block_table_stride`
        # (kernels/triton_unified_attention.py:375).
        block_table_offset = seq_idx * block_table_stride

        if q_block_offset < query_len:
            block_m_size = num_queries_per_kv * q_block_size
            kv_head_idx = tile_m.begin // num_queries_per_kv

            adjusted_tile_q_index = (
                query_start + q_block_offset + hl.arange(q_block_size)
            )
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
            max_seq_prefix_len = context_len + q_block_offset + block_m_size + 1
            max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
            num_blocks = torch.ceil(max_seq_prefix_len / page_size)
            for tile_n in hl.tile(num_blocks, block_size=num_pages_at_once):
                block_n_size = num_pages_at_once * page_size
                # explicit load due to wrong if tile_n is partial
                blk_idxs = hl.load(
                    t_block_tables,
                    [block_table_offset + tile_n.begin + hl.arange(num_pages_at_once)],
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
                S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
                S = hl.dot(q, k, out_dtype=torch.float32, acc=S) * scale
                block_m_query_mask = q_block_offset + hl.arange(
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


def helion_v12_unified_attention(
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
        # CUDA graph capture/replay: see the v9 wrapper for the rationale; the
        # buckets are frozen at capture time.
        batch_size_padded = capture_num_seqs
        max_used_querylen_padded = capture_max_query_len
        mix_num_seqs = capture_num_seqs
        mix_num_decode_tokens = capture_num_decode_tokens
        mix_max_query_len = capture_max_query_len
    else:
        max_used_querylen_padded, batch_size_padded = compute_padding(
            max_seqlen_q, num_seqs
        )
        mix_num_seqs = num_seqs
        mix_num_decode_tokens = num_decode_tokens
        mix_max_query_len = max_seqlen_q

    decode_frac_bucket, prefill_skew_bucket = _compute_mix_buckets(
        num_total_query_tokens=q.shape[0],
        num_decode_tokens=mix_num_decode_tokens,
        num_seqs=mix_num_seqs,
        max_query_len=mix_max_query_len,
    )

    # Host-side only (no GPU sync). num_seqs is the value driving the launch,
    # so the capture path is bounded by capture_num_seqs.
    assert mix_num_seqs <= MAX_NUM_SEQS, (
        f"num_seqs={mix_num_seqs} exceeds the {MAX_NUM_SEQS}-sequence bound"
    )
    num_seqs_padded = next_power_of_2(mix_num_seqs)

    kernel_helion_v12_attention(
        t_output=out,
        t_query=q,
        t_key_cache=k,
        t_value_cache=v,
        # Flattened view + explicit row stride: see the kernel body for why the
        # 2-D indexed form cannot be used with a device-computed seq_idx.
        # block_table is contiguous in all call paths, so this is a view.
        t_block_tables=block_table.reshape(-1),
        block_table_stride=block_table.stride(0),
        t_seq_lens=seqused_k,
        scale=softmax_scale,
        t_query_start_lens=cu_seqlens_q,
        num_seqs=num_seqs,
        q_block_padded_size=max_used_querylen_padded,
        batch_size_padded=batch_size_padded,
        decode_frac_bucket=decode_frac_bucket,
        prefill_skew_bucket=prefill_skew_bucket,
        num_seqs_padded=num_seqs_padded,
    )

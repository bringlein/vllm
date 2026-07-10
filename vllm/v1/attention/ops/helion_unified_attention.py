# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import helion
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
    tmp_out=None,
    tmp_L=None,
    tmp_M=None,
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
            # L init value is irrelevant: on the first tile M == -inf makes
            # alpha = exp(M - M_j) == 0, so L = L*alpha + L_j == L_j regardless.
            # Kept at 1.0 to match the Triton reference (softmax_step).
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
def kernel_helion_v10_attention(
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
    # Split-K temporary buffers (pre-allocated in backend), token-indexed and
    # mirroring Triton's softmax_segm_{output,expsum,max} triple. Each segment
    # holds an *unscaled* partial numerator / denominator / running-max so the
    # reduction stage can subtract a global max for numerical stability.
    tmp_out: torch.Tensor,  # [num_tokens, num_query_heads, num_segments, head_size]
    tmp_L: torch.Tensor,  # [num_tokens, num_query_heads, num_segments]
    tmp_M: torch.Tensor,  # [num_tokens, num_query_heads, num_segments]
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)
    # Bounded number of split-K segments per (token, head); comes from the
    # pre-allocated buffer and is independent of sequence length. Mirrors
    # Triton's NUM_PAR_SOFTMAX_SEGMENTS.
    num_segments = hl.specialize(tmp_out.size(2))

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    q_block_size = hl.register_block_size(1, q_block_padded_size)
    num_pages_at_once = hl.register_block_size(1, 32)

    # ---------------------------------------------------------------------
    # Stage 1: per-segment partial attention.
    #
    # The KV range for each (seq, query position) is split into up to
    # ``num_segments`` contiguous block ranges. Each grid tile handles one
    # segment and writes its *unscaled* online-softmax partials (numerator,
    # denominator, running max) to a unique slice of the tmp_* buffers, so no
    # cross-tile synchronization is needed within this loop.
    # ---------------------------------------------------------------------
    for seq_tile, tile_m, tile_q, tile_seg in hl.tile(
        [num_seqs, num_query_heads, max_query_len, num_segments],
        block_size=[1, num_queries_per_kv, q_block_size, 1],
    ):
        seq_idx = seq_tile.begin
        seq_len = t_seq_lens[seq_idx]
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len
        seg_idx = tile_seg.begin

        # Number of KV blocks this (seq, query position) attends to, and how
        # many blocks each segment covers so that at most ``num_segments``
        # segments are used (mirrors Triton's tiles_per_segment). Integer cdiv
        # keeps these as index-typed values (no float -> index rounding).
        max_seq_prefix_len = context_len + tile_q.begin + q_block_size + 1
        max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
        num_blocks = helion.cdiv(max_seq_prefix_len, page_size)
        blocks_per_segment = helion.cdiv(num_blocks, num_segments)

        # Absolute [start, end) block range this segment owns. When the segment
        # index is past the used range, seg_block_start >= seg_block_end, so the
        # inner loop is empty and the neutral init below is stored unchanged.
        seg_block_start = seg_idx * blocks_per_segment
        seg_block_end = torch.minimum(
            (seg_idx + 1) * blocks_per_segment, num_blocks
        )

        block_m_size = num_queries_per_kv * q_block_size
        kv_head_idx = tile_m.begin // num_queries_per_kv

        # cannot use tile_q.index directly, since tile_q.index is dynamic
        adjusted_tile_q_index = query_start + tile_q.begin + hl.arange(q_block_size)
        query_head_offset = tile_m.begin + hl.arange(num_queries_per_kv)

        # Single scalar predicate (no `and` / `else`): Helion cannot lower `and`
        # between two boolean tensors, and a divergent if/else with stores in
        # both branches is hard to lower. Instead we guard only on query
        # validity and let an empty segment fall through to the neutral store.
        if query_start + tile_q.begin < query_end:
            q_load_mask = adjusted_tile_q_index[:, None, None] < query_end
            # (tile_q, tile_m, HEAD_SIZE)
            q = hl.load(
                t_query,
                [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
                extra_mask=q_load_mask,
            )
            # (tile_m, HEAD_SIZE)
            q = q.flatten(start_dim=0, end_dim=1)

            # Neutral online-softmax state. Unlike the single-pass kernels, L
            # MUST start at 0.0 here: an empty segment runs zero inner
            # iterations, so the init survives and is stored as the neutral
            # partial (numerator 0, denominator 0, max -inf) that contributes
            # nothing to the stage-2 reduction.
            M_seg = hl.full([block_m_size], float("-inf"), dtype=torch.float32)
            L_seg = hl.full([block_m_size], 0.0, dtype=torch.float32)
            acc_seg = hl.zeros([block_m_size, head_size], dtype=torch.float32)

            # Inner sequential loop over this segment's absolute block range.
            # Empty when seg_block_start >= seg_block_end.
            for tile_n_inner in hl.tile(
                seg_block_start, seg_block_end, block_size=num_pages_at_once
            ):
                inner_block_idx = tile_n_inner.begin

                block_n_size = num_pages_at_once * page_size
                # explicit load due to wrong if tile_n is partial
                blk_idxs = hl.load(
                    t_block_tables,
                    [seq_idx, inner_block_idx + hl.arange(num_pages_at_once)],
                )
                blk_idxs = blk_idxs.view([num_pages_at_once]).to(torch.int64)
                # Compute absolute tile offsets for this inner block
                absolute_tile_token_offsets = inner_block_idx * page_size + hl.arange(
                    block_n_size
                )

                # (tile_n, PAGE_SIZE, 1, HEAD_SIZE)
                k_load = t_key_cache[blk_idxs, :, kv_head_idx, :]
                k_load = k_load.flatten(start_dim=0, end_dim=1)
                # (tile_n, HEAD_SIZE)
                k = hl.zeros([block_n_size, head_size], dtype=k_load.dtype)
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

                # Online softmax update within segment (SEQUENTIAL)
                M_j = torch.maximum(M_seg, torch.amax(S, 1))
                P = torch.exp(S - M_j[:, None])
                L_j = torch.sum(P, 1)
                alpha = torch.exp(M_seg - M_j)
                acc_seg = acc_seg * alpha[:, None]
                L_seg = (L_seg * alpha) + L_j
                M_seg = M_j

                # (tile_m, HEAD_SIZE)
                acc_seg = hl.dot(P.to(v.dtype), v, out_dtype=torch.float32, acc=acc_seg)

            # Store *unscaled* partials for this segment (single store path for
            # both active and empty segments). The reduction stage applies the
            # numerically stable exp(M_seg - overall_max) rescale. Token-indexed
            # writes to a unique (token, head, segment) slice, so no atomics /
            # cross-tile sync are required.
            store_mask = adjusted_tile_q_index[:, None] < query_end
            acc_seg_view = acc_seg.view([q_block_size, num_queries_per_kv, head_size])
            L_seg_view = L_seg.view([q_block_size, num_queries_per_kv])
            M_seg_view = M_seg.view([q_block_size, num_queries_per_kv])
            hl.store(
                tmp_out,
                [adjusted_tile_q_index, query_head_offset, seg_idx, hl.arange(head_size)],
                acc_seg_view,
                extra_mask=store_mask[:, :, None],
            )
            hl.store(
                tmp_L,
                [adjusted_tile_q_index, query_head_offset, seg_idx],
                L_seg_view,
                extra_mask=store_mask,
            )
            hl.store(
                tmp_M,
                [adjusted_tile_q_index, query_head_offset, seg_idx],
                M_seg_view,
                extra_mask=store_mask,
            )

    # Grid-wide barrier: separates the two top-level device loops so every
    # segment partial is committed before the reduction stage reads it. Helion
    # compiles this as a persistent-kernel phase boundary (host-side only).
    hl.barrier()

    # ---------------------------------------------------------------------
    # Stage 2: reduction. For each (seq, query position, head group) combine
    # the per-segment partials with a numerically stable global-max rescale
    # (mirrors Triton's reduce_segments), normalize, and write the output.
    # ---------------------------------------------------------------------
    for seq_tile, tile_m, tile_q in hl.tile(
        [num_seqs, num_query_heads, max_query_len],
        block_size=[1, num_queries_per_kv, q_block_size],
    ):
        seq_idx = seq_tile.begin
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]

        if query_start + tile_q.begin < query_end:
            adjusted_tile_q_index = query_start + tile_q.begin + hl.arange(q_block_size)
            query_head_offset = tile_m.begin + hl.arange(num_queries_per_kv)
            load_mask = adjusted_tile_q_index[:, None] < query_end

            # (q_block_size, num_queries_per_kv, num_segments)
            seg_M = hl.load(
                tmp_M,
                [adjusted_tile_q_index, query_head_offset, hl.arange(num_segments)],
                extra_mask=load_mask[:, :, None],
            )
            seg_L = hl.load(
                tmp_L,
                [adjusted_tile_q_index, query_head_offset, hl.arange(num_segments)],
                extra_mask=load_mask[:, :, None],
            )
            # (q_block_size, num_queries_per_kv, num_segments, head_size)
            seg_out = hl.load(
                tmp_out,
                [
                    adjusted_tile_q_index,
                    query_head_offset,
                    hl.arange(num_segments),
                    hl.arange(head_size),
                ],
                extra_mask=load_mask[:, :, None, None],
            )

            # Global max across segments, then stable rescale factor in [0, 1].
            overall_max = torch.amax(seg_M, dim=2)  # (q_block, nq_per_kv)
            rescale = torch.exp(seg_M - overall_max[:, :, None])  # (..., num_segments)

            overall_L = torch.sum(seg_L * rescale, dim=2)  # (q_block, nq_per_kv)
            acc = torch.sum(
                seg_out * rescale[:, :, :, None], dim=2
            )  # (q_block, nq_per_kv, head_size)

            # Safe divide (a fully-masked row has overall_L == 0).
            acc = torch.where(
                overall_L[:, :, None] > 0.0,
                acc / overall_L[:, :, None],
                acc,
            )

            store_mask = adjusted_tile_q_index[:, None, None] < query_end
            hl.store(
                t_output,
                [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
                acc,
                extra_mask=store_mask,
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
    tmp_out: torch.Tensor | None = None,
    tmp_L: torch.Tensor | None = None,
    tmp_M: torch.Tensor | None = None,
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

    # TODO: copied from triton, adjust
    # # Launch the 2D kernel if
    # # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # # 2. The batch includes at least one prefill request, or
    # # 3. The number of sequences exceeds the configured threshold, or
    # # 4. Batch invariance is enabled
    # use_3d = not (
    #     seq_threshold_3D is None
    #     or num_par_softmax_segments is None
    #     or softmax_segm_output is None
    #     or softmax_segm_max is None
    #     or softmax_segm_expsum is None
    #     or max_seqlen_q > 1
    #     or num_seqs > seq_threshold_3D
    #     or is_batch_invariant
    # )

    # Route between the 2D (v9) and 3D split-K (v10) kernels based on whether
    # the caller supplied the per-segment temporary buffers. When they are all
    # None we run the single-pass 2D kernel; otherwise we run the split-K
    # kernel that writes/reduces per-segment partials.
    use_3d = tmp_out is not None and tmp_L is not None and tmp_M is not None

    if not use_3d:
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
        return

    kernel_helion_v10_attention(
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
        tmp_out=tmp_out,
        tmp_L=tmp_L,
        tmp_M=tmp_M,
    )

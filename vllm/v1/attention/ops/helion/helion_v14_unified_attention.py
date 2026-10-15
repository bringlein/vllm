# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""helion_v14: v13's body + a finer specialization key.

Launch grid, ``gid -> (seq_idx, q_block)`` inverse and loop body are v13
verbatim (see ``helion_v13_unified_attention.py`` for the three body changes:
exp2 with prescaled ``scale``, tight causal KV bound, full/boundary KV loop
split). v14 changes only how the kernel is *specialized*, i.e. which shapes
share a tuned config. Because the bodies are identical, v14-vs-v13 is a
single-variable experiment on the key -- which is also why the config *source*
must be held equal between them; see the addendum at the end.

1. **``MAX_NUM_SEQS`` 256 -> 1024.** v13 rejected the 272..512-sequence capture
   batches vllm emits at its default ``max_num_seqs``. Only an assert ceiling:
   the reduction width is ``next_power_of_2(num_seqs)``, so no batch that
   already fit gets a wider vector. Cost of the width itself measured in
   ``.plans/v12_kernel/ablate_seqpad.py``; see ``MAX_NUM_SEQS`` below.

2. **``q_block_padded_size`` becomes a power-of-two ladder to
   ``MAX_Q_BLOCK_PADDED``.** v13 collapsed every ``max_seqlen_q >= 65`` into a
   single 128 bucket, so a 128-token prefill and an 8192-token one drew the
   same tuned config and absolute prefill length was invisible to the
   autotuner. Two consequences, both intended:

   - distinct buckets per prefill length, matching the axis the Triton
     heuristic keys on (``kernels/triton_unified_attention.py:1062-1064``, and
     ``avg_seqlen_q >= 4096`` in arXiv:2511.11581 Listing 7);
   - a higher ceiling on ``hl.register_block_size(1, q_block_padded_size)``
     (``:357``), widening the reachable ``q_block_size`` for long prefills.

   Motivation: on the v13 GB200 cache (``caches/helion_b200_v13``, 158
   entries), ~85% of the pure-prefill keys (``decode_frac_bucket == 0``) tuned
   to a 16-32 token KV tile with 1-2 warps — decode-shaped configs, which is
   what sharing a bucket with short prefills predicts.

Both changes alter the specialization key, so v14 needs its own tuning sweep;
``configs.v13_configs`` does not cover the new buckets. v9/v12/v13 are
untouched and remain separately measurable.

STATUS: work in progress. The one A/B run so far is invalid -- see below.

---

ADDENDUM: the ``.plans/v19/ab_finding1.py`` run of 2026-10-14 is CONFOUNDED and
its numbers must not be used.

That experiment tunes v13 and v14 over four prefill shapes that v13 lumps into
one bucket and v14 splits into four, to test whether the finer key pays. The
expectation recorded in the script was ~0%, since the measured optimum
(``block_m=128``, ``block_n=64``, ``w=4``) was invariant across 2k/8k/16k
prefill. What it reported instead was v14 faster by **3.8-6.6x**:

    shape       v13 arm                  v14 arm                  ratio
    pf_16x512   1.6458 ms [16,16,16] w4  0.2490 ms [16,2,4] w2    6.6x
    pf_4x2048   3.0361 ms [16,16,16] w4  0.5333 ms [16,2,4] w2    5.7x
    pf_1x8192   6.0140 ms [16,16,16] w4  1.5982 ms [16,2,4] w2    3.8x
    pf_2x4096   4.8444 ms [16,16,16] w4  0.9136 ms [16,2,4] w2    5.3x

That gap is not the key. At the time of the run v14 had
``autotuner_fn=_cached_from_cache`` **active** while v13 autotuned freely, so
the two arms differed in how they obtained a config:

- the v13 arm searched and landed on ``[16,16,16] w4`` on all four shapes --
  verbatim the config ``kernels/_autotune_guidance.py:6-8`` documents a
  default tune falling into, at 5.63x off the measured optimum. The v13 arm
  reproduced the known search failure.
- the v14 arm never searched. ``CachedFiniteSearch`` over the 158-entry
  ``configs/v13_configs.CONFIGS`` can only pick from a finite list, which
  happens to exclude the bad basin's ``block_n=256`` corner, so it was
  *protected* from the failure rather than better specialized.

The ~5x therefore measures config source, not bucket granularity -- and it
measures it backwards, since a replayed finite list beat a real search. Note
also that the ladder means v14's buckets (512/2048/4096/8192 here) have no
entry in that v13-era corpus at all, so the replay was off-key as well.

Both kernels now carry the same ``**AUTOTUNE_GUIDANCE`` and neither uses the
cache, so the comparison is single-variable again. Re-run both arms before
quoting any v13-vs-v14 number.
"""

import helion
import helion.language as hl
import torch

from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.ops.helion._autotune_guidance import AUTOTUNE_GUIDANCE
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
# program, which is what keeps the inverse cheap instead of a device binary
# search. Note this is only an assert ceiling: the actual width is
# `next_power_of_2(num_seqs)`, so raising this constant does not widen the
# reduction for any batch that already fit under the old bound.
#
# 1024 matches vLLM's default max_num_seqs on any GPU with >= 70 GiB that is not
# an A100 (vllm/engine/arg_utils.py, get_batch_defaults), which covers
# H100/H200/MI300X/B200; the 128 in SchedulerConfig.DEFAULT_MAX_NUM_SEQS only
# applies to smaller devices. The previous 256 rejected the 272..512-sequence
# capture batches that vllm's FULL_CUDA_GRAPH list contains at that setting.
#
# Cost of a wider reduction, measured by forcing the width at a fixed batch
# (.plans/v12_kernel/ablate_seqpad.py, one GB200; grid/config/data held constant,
# output bitwise-identical across widths): going 512 -> 1024 at 512 sequences
# costs 0.3% (num_warps=4) / 0.6% (num_warps=1). Growth is sub-linear and
# monotone in the width, i.e. latency that overlaps the Q load rather than an
# occupancy cliff; the mismatched 32-seq-at-1024-wide endpoint is the worst
# point measured at 2.0% and does not occur in practice.
MAX_NUM_SEQS = 1024

# Upper bound on `q_block_padded_size`, i.e. the longest query length that gets
# its own autotune bucket. Power-of-two ladder up to here, saturating after.
#
# 8192 is the ceiling, not the expected operating point: vLLM normally runs with
# chunked prefill enabled, which caps a scheduled prefill at
# `max_num_batched_tokens` (default 2048 for most configurations), so a single
# sequence's `max_seqlen_q` rarely exceeds 2k and the 4096/8192 buckets stay
# cold in practice. They exist for chunked-prefill-off / large
# `max_num_batched_tokens` deployments and for the micro-benchmarks, which drive
# `max_seqlen_q` directly.
MAX_Q_BLOCK_PADDED = 8192

# log2(e). The kernel's softmax uses exp2, so the base conversion is folded
# into `scale` on the host: exp(x) == exp2(x * log2e), and since `scale`
# multiplies the scores before the exponential this is exact up to one
# rounding of the constant. Free here because no LSE is emitted and the
# epilogue `acc / L` is base-agnostic.
LOG2E = 1.4426950408889634


def compute_padding_v14(max_seqlen_q: int, num_seqs: int) -> tuple[int, int]:
    """``compute_padding`` with both clamps raised: batch to ``MAX_NUM_SEQS``,
    query length to ``MAX_Q_BLOCK_PADDED``.

    v14-local because ``compute_padding`` is shared with v9, whose clamps must
    stay as-is to keep its caches keyed as tuned. Keeping the two batch
    constexprs on the same bound means ``batch_size_padded`` and
    ``num_seqs_padded`` stay equal on the eager CUDA path, as they were under
    the old 256 bound.

    ``q_block_padded_size`` is an unbroken power-of-two ladder up to
    ``MAX_Q_BLOCK_PADDED``, replacing v13's ``next_power_of_2(max_seqlen_q) if
    that is in [1, 8, 16, 32, 64] else 128``. Two defects of that rule:

    - every ``max_seqlen_q >= 65`` mapped to 128, so absolute prefill length
      was invisible to the autotune key and a 128-token prefill drew the same
      config as an 8192-token one;
    - ``max_seqlen_q`` in ``{2, 3, 4}`` also mapped to 128 (their
      ``next_power_of_2`` is not in the list) — harmless while 128 was the
      catch-all, wrong once it is a real long-prefill bucket.

    The ladder is the axis the Triton heuristic keys on
    (``kernels/triton_unified_attention.py:1062-1064``, and ``avg_seqlen_q >=
    4096`` in arXiv:2511.11581 Listing 7).

    Also raises the ceiling of ``hl.register_block_size(1,
    q_block_padded_size)``, widening the reachable ``q_block_size`` for long
    prefills; that is the other half of the intent. Costs more autotuning: more
    buckets, and the large end of the widened range will fail to compile
    (absorbed by ``autotune_ignore_errors=True``).
    """
    _, batch_size_padded = compute_padding(max_seqlen_q, num_seqs)
    if torch.version.cuda:
        batch_size_padded = min(MAX_NUM_SEQS, next_power_of_2(num_seqs))
    q_block_padded_size = min(MAX_Q_BLOCK_PADDED, next_power_of_2(max_seqlen_q))
    return q_block_padded_size, batch_size_padded


def _triton_baseline_fn_v14(
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
    """Autotune baseline: the Triton 2D kernel, as for v9/v13.

    v12 passes the block table flattened and dropped ``max_query_len`` from the
    kernel signature, so both are reconstructed here. The ``max_seqlen_q``
    recovery costs a device->host sync, which is fine: this runs only inside the
    autotuner, never on the hot path.

    ``scale`` arrives pre-multiplied by ``LOG2E`` (the body uses exp2), so it is
    divided back out for the base-e Triton kernel.
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
        softmax_scale=scale / LOG2E,
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
    """v14 specialization key: v9's four buckets plus ``num_seqs_padded``.

    Must stay in sync with the wrapper below — schedulers call this to compute
    the key without running the kernel.

    ``num_seqs_padded`` is a function of the ``num_seqs`` that drives the
    launch, and every branch keys at equal-or-finer granularity on it (eager
    CUDA: ``compute_padding_v14`` clamps at the same ``MAX_NUM_SEQS``; capture:
    the exact ``capture_num_seqs``), so it never splits a bucket on those paths.
    Raising ``MAX_NUM_SEQS`` past 256 therefore adds 512/1024 buckets rather
    than subdividing existing ones.

    ``q_block_padded_size``, by contrast, *does* subdivide v13's buckets: the
    ladder in ``compute_padding_v14`` splits v13's single ``>= 65 -> 128``
    bucket into 128/256/.../8192. Deliberate (that collapse is what hid prefill
    length from the autotuner), but it is why v13's caches do not transfer.
    """
    if capture_num_seqs is not None:
        batch_size_padded = capture_num_seqs
        q_block_padded_size = capture_max_query_len
        mix_num_seqs = capture_num_seqs
        mix_num_decode_tokens = capture_num_decode_tokens
        mix_max_query_len = capture_max_query_len
    else:
        q_block_padded_size, batch_size_padded = compute_padding_v14(
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

# Autotuner guidance, shared verbatim with v13 and v15
# (`kernels/_autotune_guidance.py`): a `num_warps=4` floor on the tcgen05 path
# plus seeds drawn from the measured grids in `.plans/v17/`. v14 exposes the
# same three `block_sizes` in the same order as v13/v15, so the seeds transfer
# verbatim.
#
# v14 MUST carry the same guidance as v13, because v14 exists to isolate one
# variable -- the `q_block_padded_size` ladder -- against v13
# (`.plans/v19/ab_finding1.py`). Any difference in how the two arms obtain
# their config confounds exactly that comparison; see the docstring addendum.
#
# The research repo's finite-search-from-cache autotuner is deliberately not
# ported, for the same reason as in v13 -- and worse here: that corpus was
# tuned under v13's single `>= 65 -> 128` bucket and has no entry for v14's
# 256..8192 buckets.


@helion.kernel(
    # always tuning
    # configs=CONFIGS,
    # with a cache
    # autotuner_fn=helion.from_cache(configs=CONFIGS),
    # autotuner_fn=_cached_from_cache,
    allow_warp_specialize=True,
    static_shapes=False,
    index_dtype=torch.int64,
    autotune_baseline_fn=_triton_baseline_fn_v14,
    autotune_effort="full",
    autotune_accuracy_check=False,
    autotune_ignore_errors=True,
    **AUTOTUNE_GUIDANCE,
    print_repro=False,
    print_output_code=False,
)
def kernel_helion_v14_attention(
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
    num_pages_at_once_full = hl.register_block_size(1, 32)

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
        # (kernels/triton_unified_attention.py:464).
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

            # Tight causal bound: the mask below only admits
            # kv < context_len + q_block_offset + q_block_size, so reading to
            # +block_m_size (v12) is pure masked work. Triton's equivalent is
            # kernels/triton_attention_helpers.py:191-196.
            max_seq_prefix_len = context_len + q_block_offset + q_block_size
            max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
            num_blocks = (max_seq_prefix_len + page_size - 1) // page_size

            # First kv position not visible to *every* row of this q block.
            # Tiles entirely below it need no causal mask and no seq_len
            # guard; floor to a whole tile so the boundary loop's tiles stay
            # byte-identical to v12's.
            diag_start = context_len + q_block_offset + 1
            full_pages = (
                diag_start // (num_pages_at_once_full * page_size)
            ) * num_pages_at_once_full
            full_pages = torch.minimum(full_pages, num_blocks)

            # ── full tiles: no causal mask, no seq_len zero-fill ──────────
            for tile_n in hl.tile(full_pages, block_size=num_pages_at_once_full):
                blk_idxs = hl.load(
                    t_block_tables,
                    [block_table_offset + tile_n.begin
                     + hl.arange(num_pages_at_once_full)],
                )
                blk_idxs = blk_idxs.view([num_pages_at_once_full]).to(torch.int64)
                block_n_size = num_pages_at_once_full * page_size

                k = t_key_cache[blk_idxs, :, kv_head_idx, :]
                k = k.flatten(start_dim=0, end_dim=1).transpose(0, 1)
                v = t_value_cache[blk_idxs, :, kv_head_idx, :]
                v = v.flatten(start_dim=0, end_dim=1)

                S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
                S = hl.dot(q, k, out_dtype=torch.float32, acc=S) * scale

                M_j = torch.maximum(M, torch.amax(S, 1))
                P = torch.exp2(S - M_j[:, None])
                L_j = torch.sum(P, 1)
                alpha = torch.exp2(M - M_j)
                acc = acc * alpha[:, None]
                L = (L * alpha) + L_j
                M = M_j

                acc = hl.dot(P.to(v.dtype), v, out_dtype=torch.float32, acc=acc)

            # ── boundary tiles: body as in v12 ───────────────────────────
            for tile_n in hl.tile(
                full_pages, num_blocks, block_size=num_pages_at_once
            ):
                # explicit load due to wrong if tile_n is partial
                blk_idxs = hl.load(
                    t_block_tables,
                    [block_table_offset + tile_n.begin + hl.arange(num_pages_at_once)],
                )
                blk_idxs = blk_idxs.view([num_pages_at_once]).to(torch.int64)
                block_n_size = num_pages_at_once * page_size

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
                P = torch.exp2(S - M_j[:, None])
                # (tile_m, )
                L_j = torch.sum(P, 1)
                # (tile_m, )
                alpha = torch.exp2(M - M_j)
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


def helion_v14_unified_attention(
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
        max_used_querylen_padded, batch_size_padded = compute_padding_v14(
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

    kernel_helion_v14_attention(
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
        scale=softmax_scale * LOG2E,
        t_query_start_lens=cu_seqlens_q,
        num_seqs=num_seqs,
        q_block_padded_size=max_used_querylen_padded,
        batch_size_padded=batch_size_padded,
        decode_frac_bucket=decode_frac_bucket,
        prefill_skew_bucket=prefill_skew_bucket,
        num_seqs_padded=num_seqs_padded,
    )

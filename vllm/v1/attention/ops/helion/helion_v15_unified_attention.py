# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""helion_v15: v14's specialization key + three body changes for long prefill.

Inherits v14's key (``MAX_NUM_SEQS = 1024``, power-of-two
``q_block_padded_size`` ladder to ``MAX_Q_BLOCK_PADDED``) unchanged. Three
changes to the *body*, each independently measurable and each motivated by a
measurement on the v13/v14 GB200 cache or codegen:

1. **``block_m`` decoupled from ``num_queries_per_kv``** (new
   ``head_block_size``, PINNED to ``nqpkv`` — see below). v14 tiled the head
   axis at exactly ``nqpkv``, so ``block_m = nqpkv * q_block_size`` and the
   M:N ratio was not a free variable: v14 cannot express "many query tokens,
   few heads".

   GQA head packing exists to amortize the KV load across query heads, but a
   long prefill already amortizes it across hundreds of *query tokens*, so in
   that regime the head axis buys little and costs register pressure, which
   starves N -- 85% of the pure-prefill keys in ``caches/helion_b200_v13``
   tuned to a 16-32 token KV tile with 1-2 warps, i.e. decode-shaped. That was
   the hypothesis; the measurement refuted it. ``head_block_size`` is kept as
   a named local at ``nqpkv`` (``:363``), NOT an ``hl.register_block_size``,
   so it adds no config axis and the grid/work-per-program are v14's exactly:

   - every smaller head block is slower, monotonically (1.08-5.7x prefill,
     up to 7.0x decode): the head axis is load-bearing for *coalescing*, not
     just KV reuse;
   - as a search axis it cost 1.08-1.40x tuned-vs-tuned at equal budget --
     the autotuner does pick ``head_block == nqpkv``, but the extra dimension
     dilutes a fixed budget.

   See the comment at ``:342-362`` and ``.plans/v15/REPORT.md`` §4, §7. What
   survives is the *expressibility*: the ``(kv_head, head_in_group)`` grid
   split and the head tail mask, so re-opening the axis is a one-line swap.

   The head axis is split into ``(kv_head, head_in_group)`` with dim 1 tiled
   at 1, so a sub-tile can never straddle a KV head (a partial last tile is
   masked, not aliased onto the next head's queries). Cost at the pinned
   value: one extra grid dimension, no re-reads of K/V.

   Reachable ``block_m`` at nqpkv=8 (``.plans/v15/probe_configs.py``):
   v14 ``{8,16,...,512}`` = ``8 * q_block``; an unpinned v15 would add every
   ``head_block * q_block`` product, i.e. the same maxima but any M:N ratio.
   NOTE: an earlier analysis of mine claimed the tuned shape was nqpkv=12 and
   that v14's ``block_m`` was "never a power of two". That was a misreading of
   the specialization key: the cache holds only nqpkv=8 (85 entries) and
   nqpkv=4 (73), both powers of two, and helion *requires* power-of-two tile
   extents anyway (nqpkv=12 fails with "arange's range must be a power of 2").
   The M:N-ratio argument above is what actually motivated this change.

2. **``seq_len`` guard moved from a post-load select into the load mask.**
   v14's two ``torch.where`` on ``(block_n, head_size)`` bf16 survive to
   ``arith.select`` in 197/197 cached TTGIR compiles, *after* ``local_load``
   -- the predicate is only available once the ``cp.async`` has landed, so it
   cannot suppress the memory request. Triton puts the same predicate on the
   load (``kernels/triton_unified_attention.py:574-585``). Helion's mechanism
   is ``extra_mask=`` on ``hl.load``, which is what v14 already used for Q.
   Also tightens the bound from ``seq_len`` to ``max_seq_prefix_len``.

   (Checked and *not* changed: the ``.transpose(0, 1)`` on K is already free
   -- it folds into the shared-memory layout as ``tt.trans`` into a
   ``dot_op`` operand layout in the full loop and ``ttg.memdesc_trans`` in the
   boundary loop, with ``movmatrix: 0`` in the PTX. Likewise the
   ``repeat_interleave`` is already hoisted out of both loops by LICM.)

3. **Conditional rescale: implemented, measured, and REVERTED.** ``M_j =
   maximum(M, amax(S))`` is ``>= M`` by construction, so
   ``alpha = exp2(M - M_j)`` is *exactly* 1 whenever no row's running max
   advanced, making ``acc *= alpha`` (a ``block_m x head_size`` fp32 multiply)
   and ``L *= alpha`` identities. Guarding them is numerics-neutral -- and
   measured **1.11-1.15x slower** on every shape
   (``.plans/v15/ablate_v15.py``). Two reasons, both visible in the codegen
   (``.plans/v15/v15c_output.py``):

   - the guard needs its own cross-lane reduction (``tl.sum`` 4 -> 5 per
     iteration) to make the predicate block-uniform;
   - far worse, helion must emit eight ``_copy`` shadow assignments of
     ``acc`` (128x128 fp32), ``M`` and ``L`` around the control-flow join to
     keep them loop-carried. That pins the accumulator live across the join
     and defeats the software pipeliner.

   The saved ~6k FLOPs do not pay for a reduction plus a pipelining barrier.
   The code is left unconditional; this entry is a record of the negative
   result, not a TODO.

Config shape is unchanged from v13/v14: three ``block_sizes``
(``q_block_size``, ``num_pages_at_once``, ``num_pages_at_once_full``) in the
same order, because ``head_block_size`` is pinned rather than registered. That
is what lets v15 share ``_autotune_guidance.SEED_CONFIGS`` with v13 verbatim.
The grid does gain a dimension and the body differs, so the tuned *results*
still do not transfer: v13's ``config_spec_hash`` differs from v15's
(``.plans/v17/check_seed_paths.py``), hence no ``configs=`` / cache here.
v9/v12/v13/v14 are untouched and remain separately measurable.
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


def compute_padding_v15(max_seqlen_q: int, num_seqs: int) -> tuple[int, int]:
    """``compute_padding`` with both clamps raised: batch to ``MAX_NUM_SEQS``,
    query length to ``MAX_Q_BLOCK_PADDED``. Identical to
    ``helion_v14_unified_attention.compute_padding_v14``; duplicated rather
    than imported so v15 does not depend on a WIP module.

    Version-local because ``compute_padding`` is shared with v9, whose clamps
    must stay as-is to keep its caches keyed as tuned. Keeping the two batch
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


def _triton_baseline_fn_v15(
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
    """v15 specialization key (= v14's): v9's four buckets plus
    ``num_seqs_padded``.

    Must stay in sync with the wrapper below — schedulers call this to compute
    the key without running the kernel.

    ``num_seqs_padded`` is a function of the ``num_seqs`` that drives the
    launch, and every branch keys at equal-or-finer granularity on it (eager
    CUDA: ``compute_padding_v15`` clamps at the same ``MAX_NUM_SEQS``; capture:
    the exact ``capture_num_seqs``), so it never splits a bucket on those paths.
    Raising ``MAX_NUM_SEQS`` past 256 therefore adds 512/1024 buckets rather
    than subdividing existing ones.

    ``q_block_padded_size``, by contrast, *does* subdivide v13's buckets: the
    ladder in ``compute_padding_v15`` splits v13's single ``>= 65 -> 128``
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
        q_block_padded_size, batch_size_padded = compute_padding_v15(
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

# No `configs=` / cache here: v15 keeps v13's three `block_sizes` (so the seed
# configs below transfer) but its grid has three dimensions and its body
# differs, giving a different `config_spec_hash` -- v13's 158-entry corpus does
# not transfer. Plain autotuning until a v15 cache exists; see `.plans/v15/`
# for the sweep.


# Autotuner guidance, shared verbatim with v13 (`kernels/_autotune_guidance.py`):
# a `num_warps=4` floor on the tcgen05 path plus seeds from the measured grids
# in `.plans/v17/`. Measured on this kernel: a default quick tune lands on
# `[16,16,16] w4` at 11.1140 ms; with the guidance it reaches `[16,4,4] w4` at
# 2.0708 ms while searching 15 configs instead of 64. Derivation in
# `.plans/helion_kernel_algorithm_analysis.md` addenda D-F.


@helion.kernel(
    allow_warp_specialize=True,
    static_shapes=False,
    index_dtype=torch.int64,
    autotune_baseline_fn=_triton_baseline_fn_v15,
    autotune_effort="full",
    autotune_accuracy_check=False,
    autotune_ignore_errors=True,
    **AUTOTUNE_GUIDANCE,
    print_repro=False,
    print_output_code=False,
)
def kernel_helion_v15_attention(
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
    # Head sub-tile: how many of a KV head's query heads one program covers.
    #
    # PINNED to num_queries_per_kv, i.e. v14's value, deliberately. It is a
    # plain python int, not an `hl.register_block_size`, so it does NOT add a
    # config axis. Measured reasons (.plans/v15/REPORT.md §4, §7):
    #   - every smaller value is slower, monotonically: 1.08-5.7x on prefill,
    #     up to 7.0x on decode. The head axis is load-bearing for *coalescing*,
    #     not just KV reuse -- with [token, head, head_size] layout, nqpkv=8
    #     contiguous heads is a 2 KB run per token vs 256 B for one head.
    #     Isolated at 5.66x with the KV loop made trivial
    #     (.plans/v15/diag_qload.py).
    #   - as a tunable axis it cost 1.08-1.40x tuned-vs-tuned at equal budget
    #     (.plans/v15/tune_ab_quick_v14.json vs _v15.json): the autotuner does
    #     find head_block == nqpkv in 5/5 cases, but the extra dimension
    #     dilutes a fixed search budget, so v15 tuned *worse* than v14 despite
    #     being a strict superset.
    # Kept as a named local (rather than inlining nqpkv) because the
    # (kv_head, head_in_group) grid split and the head tail mask below are the
    # reusable part: they make the trade-off expressible. To re-open it as a
    # search axis, swap in `hl.register_block_size(1, num_queries_per_kv)` --
    # correctness holds for every value, incl. non-divisors (60/60 cases).
    head_block_size = num_queries_per_kv

    # Host-side upper bound; same algebra as triton's total_num_q_blocks.
    num_q_blocks = t_query.size(0) // q_block_size + num_seqs

    # Head axis split into (kv_head, query-head-within-group) so the sub-tile
    # can never straddle a KV head: dim 1 is tiled at 1, and dim 2 ranges over
    # [0, num_queries_per_kv) so a partial last tile is *masked*, not aliased
    # onto the next KV head. At head_block_size == num_queries_per_kv this is
    # exactly v14's grid (same program count, same work per program).
    for q_blk_tile, kv_tile, head_tile in hl.tile(
        [num_q_blocks, num_kv_heads, num_queries_per_kv],
        block_size=[1, 1, head_block_size],
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
            block_m_size = head_block_size * q_block_size
            kv_head_idx = kv_tile.begin

            adjusted_tile_q_index = (
                query_start + q_block_offset + hl.arange(q_block_size)
            )
            # Absolute query-head index = kv_head * nqpkv + offset-in-group.
            head_in_group = head_tile.begin + hl.arange(head_block_size)
            query_head_offset = kv_head_idx * num_queries_per_kv + head_in_group
            # Two independent bounds now: the q-token tail (as v14) and, when
            # head_block_size does not divide num_queries_per_kv, the head tail.
            q_load_mask = (adjusted_tile_q_index[:, None, None] < query_end) & (
                head_in_group[None, :, None] < num_queries_per_kv
            )
            # (tile_q, tile_h, HEAD_SIZE)
            q = hl.load(
                t_query,
                [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
                extra_mask=q_load_mask,
            )
            # (block_m, HEAD_SIZE)
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
                # Rescale is unconditional. Guarding it on
                # `torch.sum((M_j > M).to(int32)) > 0` is numerics-neutral and
                # does skip a (block_m x head_size) fp32 multiply, but measured
                # 1.11-1.15x *slower* (.plans/v15/ablate_v15.py): the guard
                # adds a cross-lane reduction per iteration and, worse, forces
                # helion to emit `acc`/`M`/`L` shadow copies around the
                # control-flow join, which keeps the accumulator live across it
                # and defeats the software pipeliner. See .plans/v15/REPORT.md.
                alpha = torch.exp2(M - M_j)
                acc = acc * alpha[:, None]
                L = (L * alpha) + L_j
                M = M_j

                acc = hl.dot(P.to(v.dtype), v, out_dtype=torch.float32, acc=acc)

            # ── boundary tiles ───────────────────────────────────────────
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

                absolute_tile_token_offsets = tile_n.begin * page_size + hl.arange(
                    block_n_size
                )
                # Same predicate, but as a *load* mask instead of a post-load
                # select (v14 did two (block_n x head_size) bf16 torch.where,
                # which survived to 197/197 TTGIR as arith.select after
                # local_load -- the compiler cannot sink a predicate back
                # across the cp.async pipeline, so it has to be written here).
                # (num_pages, page_size) token index of each cache slot.
                page_tokens = (tile_n.begin + hl.arange(num_pages_at_once))[
                    :, None
                ] * page_size + hl.arange(page_size)[None, :]
                # Bound is the *tight causal* bound, not seq_len: every column
                # >= max_seq_prefix_len is forced to -inf by causal_mask below
                # for every row of this q block (the largest query position is
                # q_block_offset + q_block_size - 1, admitting kv up to
                # context_len + q_block_offset + q_block_size - 1), so not
                # fetching them is safe. max_seq_prefix_len is already
                # min()-ed with seq_len above.
                kv_load_mask = page_tokens[:, :, None] < max_seq_prefix_len

                # (num_pages, PAGE_SIZE, HEAD_SIZE) -> (block_n, HEAD_SIZE)
                k = hl.load(
                    t_key_cache,
                    [blk_idxs, hl.arange(page_size), kv_head_idx,
                     hl.arange(head_size)],
                    extra_mask=kv_load_mask,
                )
                k = k.flatten(start_dim=0, end_dim=1)
                # (HEAD_SIZE, block_n)
                k = k.transpose(0, 1)

                v = hl.load(
                    t_value_cache,
                    [blk_idxs, hl.arange(page_size), kv_head_idx,
                     hl.arange(head_size)],
                    extra_mask=kv_load_mask,
                )
                v = v.flatten(start_dim=0, end_dim=1)

                # (block_m, block_n)
                S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
                S = hl.dot(q, k, out_dtype=torch.float32, acc=S) * scale
                # M axis is (q_block_size, head_block_size) flattened, so the
                # query position repeats head_block_size times, not nqpkv.
                block_m_query_mask = q_block_offset + hl.arange(
                    q_block_size
                ).repeat_interleave(head_block_size, dim=0)
                # construct 2d causal mask
                causal_mask = (
                    absolute_tile_token_offsets[None, :]
                    < context_len + block_m_query_mask[:, None] + 1
                )
                S = torch.where(causal_mask, S, float("-inf"))

                # (block_m)
                M_j = torch.maximum(M, torch.amax(S, 1))
                # (block_m, block_n)
                P = torch.exp2(S - M_j[:, None])
                # (block_m, )
                L_j = torch.sum(P, 1)
                # Unconditional, as in the full-tile loop; see there.
                alpha = torch.exp2(M - M_j)
                # (block_m, HEAD_SIZE)
                acc = acc * alpha[:, None]
                L = (L * alpha) + L_j
                M = M_j

                # (block_m, HEAD_SIZE)
                acc = hl.dot(P.to(v.dtype), v, out_dtype=torch.float32, acc=acc)

            # epilogue
            acc = acc / L[:, None]
            hl.store(
                t_output,
                [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
                acc.view([q_block_size, head_block_size, head_size]),
                extra_mask=q_load_mask,
            )


def helion_v15_unified_attention(
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
        max_used_querylen_padded, batch_size_padded = compute_padding_v15(
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

    kernel_helion_v15_attention(
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

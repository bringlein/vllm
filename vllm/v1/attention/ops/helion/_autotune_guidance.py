# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared autotuner guidance for the flat q-block attention kernels.

On pure prefill the *search*, not the kernel, was the dominant cost: a default
``quick`` tune of v15 lands on ``block_sizes=[16,16,16] w4`` and runs
11.1140 ms, versus 1.9755 ms for the measured optimum -- **5.63x**. Applying the
guidance below took it to 2.0708 ms while searching 15 configs instead of 64.

Full derivation in ``.plans/helion_kernel_algorithm_analysis.md`` addenda D-F;
measured grids in ``.plans/v17/``.

Why the search fails (addendum D3): the feasible tile region is narrow --
``block_m`` is capped by TMEM columns (512/CTA) and ``block_n`` by shared
memory -- and the good region is separated from the bad one by a ridge. From
the bad corner (<=2 pages, <=2 warps) the penalty is only 2.7-4.8x, but moving
*one* axis away from it costs 10-80x, so ``PatternSearch``, which perturbs one
parameter at a time, cannot cross it.

Shared by v13 and v15 because both expose the same three ``block_sizes``
(``q_block_size``, ``num_pages_at_once``, ``num_pages_at_once_full``) in the
same order, so the seed configs transfer verbatim. v9/v12 are left alone to
keep their published numbers reproducible.
"""

from __future__ import annotations

import os

import helion
import torch

# fmt: off
# ruff-format is disabled for the rest of this file on purpose: helion's
# autotune cache key hashes inspect.getsource() of the kernel function
# (helion/autotuner/local_cache.py), so reflowing a kernel body silently
# invalidates every tuned config for it. Keeping the source byte-identical to
# the research repo (helion-aot-research/kernels/) also keeps the two diffable.

__all__ = [
    "AUTOTUNE_GUIDANCE",
    "SEED_CONFIGS",
    "WARP_OVERRIDES",
    "tcgen05_attention_path",
]


def tcgen05_attention_path() -> bool:
    """True when the accumulator lands in tensor memory (sm_100+, bf16/fp16).

    The 4-warp floor is a property of *this path only*: CUTLASS hard-codes
    ``tmem_warp_shape_mn=(4,1)`` and per-warp ``tcgen05.ld`` semantics make
    that partition uncoverable by fewer warps
    (``helion/language/memory_ops.py:1590``). These kernels really are on this
    path -- the generated PTX carries 207 ``tcgen05`` instructions with an
    explicit TMEM alloc/dealloc pair and zero ``mma.sync``/``wgmma``. Hopper
    and MI300 decompose the MMA differently, so the floor must not apply there.

    Gated on compute capability rather than ``get_cute_mma_support()`` because
    the latter lives under ``helion._compiler.cute`` (private, and it probes by
    trial-compiling a CUTLASS op). Verified equivalent on the GB200 target:
    ``cc=(10,0)``, ``tcgen05_f16bf16=True``.
    """
    if not torch.version.cuda or not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability(0)[0] != 10:  # 10 == sm_100, Blackwell
        return False
    # Mirrors helion's own gate, so turning the fused path off also drops the
    # floor (and gives an A/B switch).
    return os.environ.get("HELION_CUTE_FLASH", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


# ``num_warps=4`` is optimal in *every* regime measured, so the floor is global
# and deliberately NOT keyed on the decode fraction: a mixed prefill+decode
# batch tunes to ``block_m=128`` and takes the same hit as pure prefill, so a
# ``decode_frac_bucket == 0`` condition would miss the realistic serving case.
#
#   penalty of num_warps<=2:  prefill 4.75x | mixed 4.83x | decode 1.09-1.15x
#
# ``w=8`` measured 1.3x worse than ``w=4`` at prefill tiles: more warps than
# the t2r partition needs just fragments the tile and adds synchronisation. So
# 4 is a structural constant, not a fitted value -- which is why it holds
# across prefill, mixed and decode, and why pinning it is safe rather than a
# workload-specific bet. It also shrinks the search space, recovering coverage
# (the autotuner otherwise spends ~40% of pure-prefill and ~60% of decode
# samples at w<=2, a region that cannot win).
WARP_OVERRIDES: dict[str, object] = (
    {"num_warps": 4} if tcgen05_attention_path() else {}
)

# Seeds come from the measured grids in ``.plans/v17/``, NOT from
# ``caches/helion_b200_v13``: of its 158 entries only 36% even have ``w>=4``,
# and **0** are within reach of the measured optimum (none has ``block_m>=64``
# and ``block_n>=64`` and ``w>=4``). That cache is a record of the same search
# failure 158 times over, so seeding from it would re-inject the bad basin.
#
# block_sizes order is (q_block_size, num_pages_at_once, num_pages_at_once_full)
# and block_m = num_queries_per_kv * q_block_size, block_n = pages * page_size.
SEED_CONFIGS: list[helion.Config] = [
    # long prefill and mixed: block_m=128, block_n=64 -- measured optimum.
    helion.Config(block_sizes=[16, 4, 4], num_warps=4, num_stages=2,
                  pid_type="flat"),
    # decode: block_m=8, block_n=64 (0.8157 ms at 128x8192).
    helion.Config(block_sizes=[1, 4, 4], num_warps=4, num_stages=2,
                  pid_type="flat"),
    # decode with a longer KV tile; won at 32x4096 (0.1432 ms at 8 pages).
    helion.Config(block_sizes=[1, 8, 8], num_warps=4, num_stages=2,
                  pid_type="flat"),
    # short prefill, where block_n=64 is not yet amortised.
    helion.Config(block_sizes=[16, 2, 2], num_warps=4, num_stages=2,
                  pid_type="flat"),
    # second best at the prefill tile; hedge in case num_stages shifts it.
    helion.Config(block_sizes=[16, 4, 4], num_warps=8, num_stages=2,
                  pid_type="flat"),
]

# Spread as ``**AUTOTUNE_GUIDANCE`` into ``@helion.kernel(...)``.
#
# Note that ``FROM_BEST_AVAILABLE`` combines *two independent* sources, which
# is easy to conflate (``base_search.py:939-1048``):
#
#   1. explicit ``autotune_seed_configs`` + compiler seeds + the default
#      config. These are what we actually want, and they are pinned finalists.
#   2. up to ``autotune_best_available_max_configs`` configs recovered from
#      ``HELION_CACHE_DIR`` by ``_find_similar_cached_configs``.
#
# Source 2 is the genuine warm-start mechanism and is *desirable* in steady
# state: re-tuning a shape that was already tuned well should start from that
# result. It is only harmful when the cache contains a bad corpus -- as the
# v13-era ``caches/helion_b200_v13`` does (0/158 entries near the measured
# optimum). But that is a property of one specific directory, not of the
# mechanism, and it is already isolated: cached configs are filtered on
# hardware + specialization_key + ``config_spec_hash``, and v13/v15 have
# *different* structural fingerprints (measured in
# ``.plans/v17/check_seed_paths.py``: 0dedcc5f... vs 1a084605...), so the old
# corpus cannot transfer into either kernel as configured today.
#
# So we do NOT throttle source 2. Suppress it per-run instead, only when
# tuning against a cache dir known to hold a bad corpus:
#     HELION_SKIP_CACHE=1          # drops cached configs, keeps explicit seeds
#     HELION_CACHE_DIR=<fresh dir> # or simply tune somewhere clean
AUTOTUNE_GUIDANCE: dict[str, object] = {
    # Deletes the unwinnable low-warp region outright.
    "autotune_config_overrides": WARP_OVERRIDES,
    # Start the population inside the good basin instead of hoping to reach it.
    "autotune_seed_configs": SEED_CONFIGS,
    # Required for ``autotune_seed_configs`` to be read at all: seeds are
    # consumed only by ``_generate_best_available_population_flat``
    # (``helion/autotuner/base_search.py:939``), which runs under
    # FROM_BEST_AVAILABLE. The default is FROM_RANDOM, so without this the
    # seeds are a silent no-op. Set here rather than via
    # HELION_AUTOTUNER_INITIAL_POPULATION so there is no env dependency; the
    # kernel setting takes precedence over both the env var and the effort
    # profile.
    "autotune_initial_population_strategy": "from_best_available",
}

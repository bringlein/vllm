# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Helion paged-attention kernel variants.

Every variant exposes the same keyword signature, so the backend calls
``helion_unified_attention`` and this module resolves which one to run from
``VLLM_HELION_ATTN_VERSION``. The variable is intentionally *not* registered in
``vllm/envs.py``: these kernels are experimental and the selection is meant to
be made explicitly per run, never defaulted.
"""

import os
from collections.abc import Callable

HELION_ATTN_VERSION_ENV = "VLLM_HELION_ATTN_VERSION"

# Maps the env value to (module, wrapper function) within this package.
_VARIANTS: dict[str, tuple[str, str]] = {
    "v5": ("helion_v5_unified_attention", "helion_v5_unified_attention"),
    "v9": ("helion_unified_attention", "helion_unified_attention"),
    "v12": ("helion_v12_unified_attention", "helion_v12_unified_attention"),
    "v13": ("helion_v13_unified_attention", "helion_v13_unified_attention"),
    "v14": ("helion_v14_unified_attention", "helion_v14_unified_attention"),
    "v15": ("helion_v15_unified_attention", "helion_v15_unified_attention"),
}


def get_helion_attn_version() -> str:
    """Resolve the selected variant, erroring out when unset or unknown."""
    version = os.environ.get(HELION_ATTN_VERSION_ENV)
    if not version:
        raise RuntimeError(
            f"{HELION_ATTN_VERSION_ENV} must be set to one of "
            f"{sorted(_VARIANTS)} to use EXPERIMENTAL_HELION_ATTN."
        )
    version = version.strip().lower()
    if version not in _VARIANTS:
        raise RuntimeError(
            f"{HELION_ATTN_VERSION_ENV}={version!r} is not a known Helion "
            f"attention kernel; expected one of {sorted(_VARIANTS)}."
        )
    return version


def get_helion_unified_attention() -> Callable[..., None]:
    """Import and return the selected variant's wrapper.

    Imported lazily so a vLLM build without helion installed only fails when
    the backend is actually selected.
    """
    module_name, fn_name = _VARIANTS[get_helion_attn_version()]
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, fn_name)

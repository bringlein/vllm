# SPDX-License-Identifier: Apache-2.0
"""
This file is a (slightly adapted) copy of https://github.com/IBM/triton-dejavu/blob/main/triton_dejavu/jit_cache.py (copied here to reduce external dependencies).

The launch overhead of triton kernels is a well known problem (see e.g. [1](https://github.com/triton-lang/triton/pull/3503), [2](https://github.com/triton-lang/triton/issues/2637), [3](https://github.com/triton-lang/triton/issues/6064)). Parts of the launch overhead comes from the fact that the triton JIT checks very carefully if an existing binary is safe to use. 

In many scenarios, these checks can be relaxed. 
This PR adds such a cache with relaxed checks is implemented by `triton_dejavu.jitcache`. It is implemented as a decorator that could be used in front of the `triton.jit` decorator: 

```
@triton_dejavu.jitcache(
    check_keys=["USE_ALIBI_SLOPES", "SLIDING_WINDOW", "filter_by_query_len"],
)
@triton.jit
def kernel_paged_attention_.... 
```

The required `check_keys` argument must provide a list of the kernel parameters marked as `tl.constexpr` that **must be checked** to select the correct kernel binary. Ideally, this is just a subset of all constant kernel parameters.
For example, if we have two constant parameters A and B, but we know that A never will change in a particular application, but B will, then the list should look like `check_keys=["A"]`.

Consequently, *the usage of `triton_dejavu.jitcache` is application specific* (and also *experimental*).

Additionally, a user could provide a lock with e.g. `cache_lock=triton_dejavu.global_cache_lock` to ensure that no re-compilation happens after the cache lock is locked.

The `triton_dejavu.jitcache` reduces the launch overhead of triton kernels to 30-40 micro-seconds.

Details see: https://github.com/IBM/triton-dejavu

Authors:
- Burkhard Ringlein <ngl@zurich.ibm.com>

"""

from __future__ import annotations

import sys
import os
import time
import inspect

from triton import KernelInterface
from triton.runtime.driver import driver
from triton.runtime.autotuner import OutOfResources
from triton import __version__ as triton_version

triton_major_version = int(triton_version.split(".")[0])
triton_minor_version = int(triton_version.split(".")[1])
triton_version_float = triton_major_version + float(triton_minor_version / 10)

from vllm.logger import init_logger
logger = init_logger(__name__)


__print_name__ = "vllm.triton_utils.jit_cache"


class CacheLock:

    def __init__(self, id="unkown"):
        self.is_locked = False
        self.id = id

    def lock(self):
        self.is_locked = True
        logger.debug(f"JitCache lock '{self.id}' is LOCKED.")

    def unlock(self):
        self.is_locked = False
        logger.debug(f"JitCache lock '{self.id}' is UNLOCKED.")

# to provide a global lock
global_cache_lock = CacheLock("global")


class PreparedKernel:
    def __init__(
        self,
        grid_obj,
        grid_example,
        cache_launch_grid,
        kernel,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        non_const_arg_names,
        cache_key,
        device,
        stream,
    ):
        self.grid_obj = grid_obj
        self.grid_is_callable = callable(grid_obj)
        self.grid_size = len(
            grid_example
        )  # grid_example is always not callable, so we need both
        self.cache_launch_grid = cache_launch_grid
        self.concrete_grid = None
        if cache_launch_grid:
            grid_0 = grid_example[0]
            grid_1 = grid_example[1] if self.grid_size > 1 else 1
            grid_2 = grid_example[2] if self.grid_size > 2 else 1
            self.concrete_grid = (grid_0, grid_1, grid_2)
        self.kernel = kernel
        self.launch_metadata = launch_metadata
        self.launch_enter_hook = launch_enter_hook
        self.launch_exit_hook = launch_exit_hook
        self.non_const_arg_names = non_const_arg_names

        # TODO: safe to cache?
        self.device = device
        self.stream = stream
        self._init_handles()
        self.cache_key = cache_key

    def _init_handles(self):
        """
        more or less redo what CompiledKernel._init_hanles is doing
        (c.f. triton/python/triton/runtime/compiler.py:379)
        """
        self.run = driver.active.launcher_cls(self.kernel.src, self.kernel.metadata)
        # check once and not again
        self.dev_max_shared = driver.active.utils.get_device_properties(self.device)[
            "max_shared_mem"
        ]
        if self.kernel.metadata.shared > self.dev_max_shared:
            raise OutOfResources(
                self.metadata.shared, self.dev_max_shared, "shared memory"
            )
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills = (
            driver.active.utils.load_binary(
                self.kernel.name,
                self.kernel.kernel,
                self.kernel.metadata.shared,
                self.device,
            )
        )

    def __call__(self, *args, **kwargs):
        assert len(args) == 0

        non_constsexpr_vals = []
        # order is always the same...
        for arg_n in self.non_const_arg_names:
            non_constsexpr_vals.append(kwargs[arg_n])

        if self.cache_launch_grid:
            grid_0, grid_1, grid_2 = self.concrete_grid
        else:
            if self.grid_is_callable:
                grid = kwargs["grid"](kwargs)
            else:
                grid = kwargs["grid"]
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1

        return self.run(
            grid_0,
            grid_1,
            grid_2,
            self.stream,
            self.function,
            self.kernel.packed_metadata,
            self.launch_metadata,
            self.launch_enter_hook,
            self.launch_exit_hook,
            *non_constsexpr_vals,
        )

    def get_key(self):
        return self.cache_key


class JitCache(KernelInterface):

    def __init__(
        self,
        fn,
        arg_names,
        check_keys,
        cache_lock: CacheLock,
        cache_launch_grid=False,
    ):
        assert 3.0 <= triton_version_float <= 3.2
        self.arg_names = arg_names
        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn
        self.cache_lock = cache_lock
        self.cache_launch_grid = cache_launch_grid
        self.dynamic_mode = False
        self.run = self._run_static
        if self.cache_lock is None:
            self.dynamic_mode = True
            self.run = self._run_dynamic
        self.check_keys = check_keys
        self.kernel_cache = {}

        def calc_cache_index(kwargs):
            cache_key = ""
            for c_arg_name in check_keys:
                cache_key += str(kwargs[c_arg_name])
            return cache_key

        self.cache_index_func = calc_cache_index
        if len(check_keys) == 0:
            self.cache_index_func = lambda ignore: "_default_"

    def _get_prepared_kernel(self, *args, **kwargs) -> PreparedKernel:
        """
        more or less redo what JITFunction.run is doing
        (c.f. triton/python/triton/runtime/jit.py:565)
        """

        kwargs["warmup"] = True
        compile_start = time.time()
        kernel = self.fn.run(*args, **kwargs)
        compile_end = time.time()

        const_arg_names = []
        non_const_arg_names = []
        for p in self.fn.params:
            if p.is_constexpr or p.is_const:
                const_arg_names.append(p.name)
            else:
                non_const_arg_names.append(p.name)
        if any(x in self.check_keys for x in non_const_arg_names):
            raise RuntimeError(
                f"[{__print_name__}] ERROR: check_keys must only contain parameters marked as tl.constexpr (non-constants will be updated in all cases)."
            )

        (
            bound_args,
            sig_and_spec,
            constexpr_vals,
            non_constexpr_vals,
            excess_kwargs,
        ) = self.fn.binder(*args, **kwargs)
        bind_end = time.time()

        if callable(kwargs["grid"]):
            grid = kwargs["grid"](kwargs)
        else:
            grid = kwargs["grid"]

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)

        prepared_kernel = PreparedKernel(
            kwargs["grid"],
            grid,
            self.cache_launch_grid,
            kernel,
            launch_metadata,
            self.fn.CompiledKernel.launch_enter_hook,
            self.fn.CompiledKernel.launch_exit_hook,
            non_const_arg_names,
            self.cache_index_func(kwargs),
            device,
            stream,
        )

        wrapper_end = time.time()
        compile_time = compile_end - compile_start
        bind_time = bind_end - compile_end
        wrapper_time = wrapper_end - bind_end

        logger.debug(f"JIT compilation took {compile_time}s, binding {bind_time}, wrapper {wrapper_time}s.")

        return prepared_kernel

    def _run_static(self, *args, **kwargs):
        # we only support kwargs
        if len(args) != 0:
            raise RuntimeError(
                f"[{__print_name__}] ERROR: The JITCache only supports kwargs, len(args) must be 0."
            )
        # assert no config pre-hook
        assert "pre_hook" not in kwargs or kwargs["pre_hook"] is None

        # print(f"my lock: {self.cache_lock.is_locked}")
        if not self.cache_lock.is_locked:
            # we only support int, bool, float as cache index
            for key in self.check_keys:
                assert type(kwargs[key]) in [int, bool, float]
            prepared_kernel = self._get_prepared_kernel(*args, **kwargs)
            if prepared_kernel.get_key() in self.kernel_cache: 
                logger.debug(
                    f"WARNING: Kernel variant already cached, will override (cache lock is not locked). "
                    f"This could mean that the given check_keys are ambigous (or the same call was already executed)."
                )
            self.kernel_cache[prepared_kernel.get_key()] = prepared_kernel

        try:
            kernel_variant = self.kernel_cache[self.cache_index_func(kwargs)]
        except KeyError as e:
            print(
                f"[{__print_name__}] ERROR: Key {self.cache_index_func(kwargs)}  not in cache.\n"
                f"Current cache: {list(self.kernel_cache.keys())}"
            )
            print(e)
            raise e

        return kernel_variant(*args, **kwargs)

    def _run_dynamic(self, *args, **kwargs):
        # we only support kwargs
        if len(args) != 0:
            raise RuntimeError(
                f"[{__print_name__}] ERROR: The JITCache only supports kwargs, len(args) must be 0."
            )
        # assert no config pre-hook
        assert "pre_hook" not in kwargs or kwargs["pre_hook"] is None

        try:
            kernel_variant = self.kernel_cache[self.cache_index_func(kwargs)]
        except KeyError as e:
            logger.debug(
                f"Key {self.cache_index_func(kwargs)}  not in cache, compiling...\n"
                f"Current cache: {list(self.kernel_cache.keys())}"
            )
            # we only support int, bool, float as cache index
            for key in self.check_keys:
                assert type(kwargs[key]) in [int, bool, float]
            kernel_variant = self._get_prepared_kernel(*args, **kwargs)
            self.kernel_cache[kernel_variant.get_key()] = kernel_variant

        return kernel_variant(*args, **kwargs)


def jitcache(
    check_keys,
    cache_lock=None,
    cache_launch_grid=False,
):
    """
    Decorator for caching a :code:`triton.jit`'d function.

    :param check_keys: The list of tl.constexpr that are used to index the cache. Only types int, bool, float are supported.
    :type check_keys: list[str]
    :param cache_lock: The CacheLock used for this JitCache.
    :type cache_lock: CacheLock
    :param chache_launch_grid: Indicate if the launch grid size is static and should be cached (False by default).
    :type cache_launch_grid: bool
    """

    def decorator(fn):
        return JitCache(
            fn,
            fn.arg_names,
            check_keys,
            cache_lock,
            cache_launch_grid,
        )

    return decorator

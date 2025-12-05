# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import helion
import helion.language as hl
import torch

from .triton_unified_attention import (
    unified_attention as triton_baseline_unified_attention,
)


def _triton_baseline_fn(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    # k_scale,
    # v_scale,
    t_query_start_lens,  # [num_seqs+1]
    max_query_len,
    max_seq_len,
    # max_used_query_len_padded,
    num_seqs,
    # is_decode_only,
):
    # max_seq_len = t_seq_lens.max()
    # max_query_len = t_query_start_lens.diff().max()
    return triton_baseline_unified_attention(
        q=t_query,
        k=t_key_cache,
        v=t_value_cache,
        out=t_output,
        cu_seqlens_q=t_query_start_lens,
        max_seqlen_q=max_query_len,
        seqused_k=t_seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=t_block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

# config = nv_config if torch.version.cuda else amd_config

# dbg_config = helion.Config(block_sizes=[2, 4], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', 'last', 'last', 'first', 'first', 'last', 'first', '', 'last'], loop_orders=[[1, 2, 0]], num_stages=2, num_warps=4, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, True], range_num_stages=[0, 2], range_unroll_factors=[0, 1], range_warp_specializes=[])
# TODO: bug, it does not use tl.program_id(1) if changing to xyz...? 
# # dbg_config = helion.Config(block_sizes=[2, 4], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', 'last', 'last', 'first', 'first', 'last', 'first', '', 'last'], loop_orders=[[1, 2, 0]], num_stages=2, num_warps=4, pid_type='xyz', range_flattens=[None, False], range_multi_buffers=[None, True], range_num_stages=[0, 2], range_unroll_factors=[0, 1], range_warp_specializes=[])
# dbg_config = helion.Config(block_sizes=[4, 2], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', 'first', 'last', 'first', 'first', 'last', 'first', '', 'last'], loop_orders=[[1, 2, 0]], num_stages=1, num_warps=4, pid_type='persistent_interleaved', range_flattens=[True, False], range_multi_buffers=[True, True], range_num_stages=[1, 2], range_unroll_factors=[0, 1], range_warp_specializes=[])
# dbg_config = helion.Config(block_sizes=[1], indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[64], load_eviction_policies=['last', 'last', 'first', 'last', 'first', '', '', '', 'last', 'last', ''], loop_orders=[[2, 1, 0]], num_stages=1, num_warps=1, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 2], range_unroll_factors=[0, 0], range_warp_specializes=[])
#dbg_config = helion.Config(block_sizes=[1, 1], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', 'last', 'last', 'first', 'first', 'last', 'first', '', 'last'], loop_orders=[[1, 2, 0]], num_stages=2, num_warps=4, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, True], range_num_stages=[0, 2], range_unroll_factors=[0, 1], range_warp_specializes=[])
# dbg_config = helion.Config(block_sizes=[8, 2], indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], l2_groupings=[8], load_eviction_policies=['last', 'first', 'last', 'last', 'last', 'last', 'last'], loop_orders=[[1, 0, 2]], num_stages=5, num_warps=4, pid_type='persistent_interleaved', range_flattens=[True, None], range_multi_buffers=[True, False], range_num_stages=[1, 1], range_unroll_factors=[2, 1], range_warp_specializes=[])

# dbg_config = helion.Config(block_sizes=[32, 2], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], l2_groupings=[1], load_eviction_policies=['', 'first', 'last', 'last', 'last', 'first', ''], loop_orders=[[0, 1]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, None, False], range_multi_buffers=[None, True, False], range_num_stages=[], range_unroll_factors=[0, 2, 1], range_warp_specializes=[])
#dbg_config = helion.Config(block_sizes=[8, 32, 1, 2], indexing=['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[8], load_eviction_policies=['last', 'last', 'first', 'last', 'first', '', ''], loop_orders=[[1, 0]], num_stages=5, num_warps=4, pid_type='persistent_interleaved', range_flattens=[None, False, True], range_multi_buffers=[True, True, False], range_unroll_factors=[2, 3, 1], range_warp_specializes=[])

# dbg_config = helion.Config(block_sizes=[32, 2], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], l2_groupings=[2], load_eviction_policies=['last', 'first', 'last', 'last', 'last', 'last', 'last'], loop_orders=[[1, 0]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, False, None], range_multi_buffers=[None, False, True], range_num_stages=[], range_unroll_factors=[0, 1, 2], range_warp_specializes=[])
dbg_configs = [
    helion.Config(block_sizes=[32, 2], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], l2_groupings=[2], load_eviction_policies=['last', 'first', 'last', 'last', 'last', 'last', 'last'], loop_orders=[[1, 0]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, False, None], range_multi_buffers=[None, False, True], range_num_stages=[], range_unroll_factors=[0, 1, 2], range_warp_specializes=[]),
    helion.Config(block_sizes=[1, 2], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], l2_groupings=[2], load_eviction_policies=['last', 'first', 'last', 'last', 'last', 'last', 'last'], loop_orders=[[1, 0]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, False, None], range_multi_buffers=[None, False, True], range_num_stages=[], range_unroll_factors=[0, 1, 2], range_warp_specializes=[]),
    helion.Config(block_sizes=[32, 1], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], l2_groupings=[2], load_eviction_policies=['last', 'first', 'last', 'last', 'last', 'last', 'last'], loop_orders=[[1, 0]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, False, None], range_multi_buffers=[None, False, True], range_num_stages=[], range_unroll_factors=[0, 1, 1], range_warp_specializes=[]),
    helion.Config(block_sizes=[1, 1], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], l2_groupings=[2], load_eviction_policies=['last', 'first', 'last', 'last', 'last', 'last', 'last'], loop_orders=[[1, 0]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, False, None], range_multi_buffers=[None, False, True], range_num_stages=[], range_unroll_factors=[0, 1, 1], range_warp_specializes=[]),
    ]
dbg_config = helion.Config(block_sizes=[16, 2], indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], l2_groupings=[32], load_eviction_policies=['first', '', 'last', 'last', 'last', '', 'last'], loop_orders=[[0, 1]], num_stages=4, num_warps=4, pid_type='flat', range_flattens=[None, True, False], range_multi_buffers=[None, True, None], range_num_stages=[], range_unroll_factors=[0, 0, 1], range_warp_specializes=[])

static_shapes = True


@helion.kernel(
    allow_warp_specialize=True,
    # static_shapes=False,
    # static_shapes=True,
    static_shapes=static_shapes,
    # dot_precision='ieee',
    # config=config,
    config=dbg_config,
    # configs=nv_configs,
    # configs=dbg_configs,
    autotune_baseline_fn=_triton_baseline_fn,
    # # autotune_accuracy_check=False,  # since we can't adjust ATOL manually
    autotune_effort="quick",
    # autotune_ignore_errors=True,
    print_output_code=False,
    print_repro=False,
    # debug_dtype_asserts=True,
    index_dtype=torch.int64,
)
def kernel_helion_v4_attention(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    # k_scale,
    # v_scale,
    t_query_start_lens,  # [num_seqs+1]
    max_query_len,  # must be on cpu
    max_seq_len,  # must be on cpu
    # max_used_query_len_padded: hl.constexpr,
    num_seqs,  # must be on cpu
    # to trigger re-compilation for decode only
    # is_decode_only: hl.constexpr,
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    # q_block_size = hl.register_block_size(1, int(max_query_len))
    # q_block_size = hl.register_block_size(1, 32)
    # q_block_size = hl.register_block_size(1, torch.minimum(32, max_query_len))
    # if is_decode_only:
    #     q_block_size = hl.register_block_size(1, 1)
    # q_block_size = hl.register_block_size(1, 32) if not is_decode_only else hl.register_block_size(1, 1)
    # q_block_size = hl.register_block_size(1, 1)
    q_block_size = hl.register_block_size(1, min(32, max_query_len))
    # q_block_size = 1
    # q_block_size = hl.register_block_size(1, int(max_used_query_len_padded))
    # num_pages_at_once = hl.register_block_size(1, 512//page_size)
    num_pages_at_once = hl.register_block_size(1, 32)
    # num_pages_at_once = hl.register_block_size(1, torch.minimum(32, max_seq_len//page_size))
    # num_pages_at_once = hl.register_block_size(1, 1)

    for seq_tile, tile_m in hl.tile(
        [num_seqs, num_query_heads],
        block_size=[1, num_queries_per_kv],
    ):
        seq_idx = seq_tile.begin # is scalar
        seq_len = t_seq_lens[seq_idx]
        # TODO: return if seq_len == 0? How does it work with cudagraphs? 
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len

        for tile_q in hl.tile(query_start, query_end, block_size=q_block_size):
            block_m_size = tile_m.block_size * tile_q.block_size
            kv_head_idx = tile_m.begin // num_queries_per_kv

            # (tile_q, tile_m, HEAD_SIZE)
            q = t_query[tile_q, tile_m, :]
            # (tile_m, HEAD_SIZE)
            q = q.flatten(start_dim=0, end_dim=1)
            # DEBUG: check the dimensions
            # q = q.view([block_m_size, head_size])

            M = hl.full(
                [block_m_size], float("-inf"), dtype=torch.float32
            )
            L = hl.full([block_m_size], 1.0, dtype=torch.float32)
            acc = hl.zeros(
                [block_m_size, head_size], dtype=torch.float32
            )

            num_blocks = torch.ceil(seq_len / page_size)
            # adjust for causal mask
            # max_seq_prefix_len = (
            #     context_len
            #     # + tile_q.end
            #     + cur_qblock_end
            #     + (tile_m.block_size + num_queries_per_kv - 1) // num_queries_per_kv
            # )
            # max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
            # num_blocks = torch.ceil(max_seq_prefix_len / page_size)
            for tile_n in hl.tile(num_blocks, block_size=num_pages_at_once):
                block_n_size = tile_n.block_size * page_size
                blk_idxs = t_block_tables[seq_idx, tile_n]
                # blk_idxs = blk_idxs.view([num_pages_at_once]).to(torch.int64)
                blk_idxs = blk_idxs.flatten().to(torch.int64)

                k_load = t_key_cache[blk_idxs, :, kv_head_idx, :]
                # DEBUG: to assert shape
                # k_load = k_load.view([tile_n, page_size, head_size])
                k_load = k_load.flatten(start_dim=0, end_dim=1)
                # DEBUG: check the dimensions
                # k_load = k_load.view([block_n_size, head_size])
                # (tile_n, PAGE_SIZE, HEAD_SIZE)
                v_load = t_value_cache[blk_idxs, :, kv_head_idx, :]
                v_load = v_load.flatten(start_dim=0, end_dim=1)
                # DEBUG: check the dimensions
                # v_load = v_load.view([block_n_size, head_size])
                
                # (tile_n, HEAD_SIZE)
                k = hl.zeros([block_n_size, head_size], dtype=k_load.dtype)
                # k = hl.full([block_n_size, head_size],float("-inf"), dtype=k_load.dtype)
                tile_token_offsets = hl.arange(block_n_size)
                absolute_tile_token_offsets = tile_n.begin * block_n_size + tile_token_offsets
                k = torch.where(absolute_tile_token_offsets[:, None] < seq_len, 
                                k_load, k)
                # (HEAD_SIZE, tile_n)
                k = k.transpose(0, 1)
                # (tile_n, HEAD_SIZE)
                v = hl.zeros([block_n_size, head_size], dtype=v_load.dtype)
                # v = hl.full([block_n_size, head_size],float("-inf"), dtype=v_load.dtype)
                v = torch.where(absolute_tile_token_offsets[:, None] < seq_len, 
                                v_load, v)
                # (tile_m, tile_n)
                S = hl.dot(q, k, out_dtype=torch.float32) * scale
                # DEBUG: to check the shape...
                S = S.view([block_m_size, block_n_size])
                # all query heads for one query token are valid
                # block_m_query_mask = tile_q.begin + hl.arange(block_m_size) // num_queries_per_kv
                # block_m_query_mask = tile_q.index.repeat_interleave(tile_m.block_size, dim=0, output_size=block_m_size)
                # block_m_query_mask = tile_q.index.repeat_interleave(tile_m.block_size, dim=0, output_size=S.size(0))
                # block_m_query_mask = tile_q.index.repeat_interleave(tile_m.block_size, dim=0)
                block_m_query_mask = tile_q.index.repeat_interleave(tile_m.block_size, dim=0) - query_start
                # DEBUG: to check the shape...
                block_m_query_mask = block_m_query_mask.view([block_m_size])
                # construct 2d causal mask
                causal_mask = absolute_tile_token_offsets[None, :] < context_len + block_m_query_mask[:, None] + 1
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
            t_output[tile_q, tile_m, :] = acc.view([tile_q.block_size, tile_m.block_size, head_size])


decode_dbg_configs = [
    helion.Config(block_sizes=[2], indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], l2_groupings=[16], load_eviction_policies=['first', 'first', 'last', '', 'last'], loop_orders=[[0, 1]], num_stages=4, num_warps=4, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, False], range_num_stages=[], range_unroll_factors=[0, 0], range_warp_specializes=[]),
    helion.Config(block_sizes=[1], indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], l2_groupings=[16], load_eviction_policies=['first', 'first', 'last', '', 'last'], loop_orders=[[0, 1]], num_stages=4, num_warps=4, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, False], range_num_stages=[], range_unroll_factors=[0, 0], range_warp_specializes=[]),
]
decode_dbg_config = helion.Config(block_sizes=[2], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], l2_groupings=[64], load_eviction_policies=['last', 'last', 'last', 'last', 'last'], loop_orders=[[1, 0]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, False], range_num_stages=[], range_unroll_factors=[0, 1], range_warp_specializes=[])


@helion.kernel(
    allow_warp_specialize=True,
    # static_shapes=False,
    # static_shapes=True,
    static_shapes=static_shapes,
    # dot_precision='ieee',
    # config=config,
    # config=decode_dbg_config,
    # configs=nv_configs,
    # configs=dbg_configs,
    autotune_baseline_fn=_triton_baseline_fn,
    # # autotune_accuracy_check=False, 
    # autotune_baseline_atol=0.3,
    autotune_effort="quick",
    # autotune_ignore_errors=True,
    print_output_code=False,
    print_repro=False,
    # debug_dtype_asserts=True,
    index_dtype=torch.int64,
)
def kernel_helion_v4_attention_decode_only(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    # k_scale,
    # v_scale,
    t_query_start_lens,  # [num_seqs+1]
    max_query_len,  # must be on cpu
    max_seq_len,  # must be on cpu
    # max_used_query_len_padded: hl.constexpr,
    num_seqs,  # must be on cpu
    # to trigger re-compilation for decode only
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    # q_block_size = hl.register_block_size(1, int(max_used_query_len_padded))
    # num_pages_at_once = hl.register_block_size(1, 512//page_size)
    num_pages_at_once = hl.register_block_size(1, 32)
    # num_pages_at_once = hl.register_block_size(1, torch.minimum(32, max_seq_len//page_size))
    # num_pages_at_once = hl.register_block_size(1, 1)

    for seq_tile, tile_m in hl.tile(
        [num_seqs, num_query_heads],
        block_size=[1, num_queries_per_kv],
    ):
        seq_idx = seq_tile.begin # is scalar
        seq_len = t_seq_lens[seq_idx]
        # TODO: return if seq_len == 0? How does it work with cudagraphs? 
        # query_start = t_query_start_lens[seq_idx]
        # query_end = t_query_start_lens[seq_idx + 1]
        # query_len = query_end - query_start
        query_len = 1
        context_len = seq_len - query_len
        
        kv_head_idx = tile_m.begin // num_queries_per_kv
        block_m_size = tile_m.block_size
        # we have to do * 1, to make tracing of symbolic shapes correct
        # block_m_size = tile_m.block_size * seq_tile.block_size

        # (tile_q, tile_m, HEAD_SIZE)
        # q = t_query[tile_q, tile_m, :]
        # q = t_query[query_start, tile_m, :]
        # q = t_query[seq_tile, tile_m, :]
        q = t_query[seq_idx, tile_m, :]
        # (tile_m, HEAD_SIZE)
        q = q.flatten(start_dim=0, end_dim=1)
        # DEBUG: check the dimensions
        q = q.view([block_m_size, head_size])

        M = hl.full(
            [block_m_size], float("-inf"), dtype=torch.float32
        )
        L = hl.full([block_m_size], 1.0, dtype=torch.float32)
        acc = hl.zeros(
            [block_m_size, head_size], dtype=torch.float32
        )

        num_blocks = torch.ceil(seq_len / page_size)
        # adjust for causal mask
        # max_seq_prefix_len = (
        #     context_len
        #     # + tile_q.end
        #     + cur_qblock_end
        #     + (tile_m.block_size + num_queries_per_kv - 1) // num_queries_per_kv
        # )
        # max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
        # num_blocks = torch.ceil(max_seq_prefix_len / page_size)
        for tile_n in hl.tile(num_blocks, block_size=num_pages_at_once):
            block_n_size = tile_n.block_size * page_size
            blk_idxs = t_block_tables[seq_idx, tile_n]
            # blk_idxs = blk_idxs.view([num_pages_at_once]).to(torch.int64)
            blk_idxs = blk_idxs.flatten().to(torch.int64)

            k_load = t_key_cache[blk_idxs, :, kv_head_idx, :]
            # DEBUG: to assert shape
            # k_load = k_load.view([tile_n, page_size, head_size])
            k_load = k_load.flatten(start_dim=0, end_dim=1)
            # DEBUG: check the dimensions
            # k_load = k_load.view([block_n_size, head_size])
            # (tile_n, PAGE_SIZE, HEAD_SIZE)
            v_load = t_value_cache[blk_idxs, :, kv_head_idx, :]
            v_load = v_load.flatten(start_dim=0, end_dim=1)
            # DEBUG: check the dimensions
            # v_load = v_load.view([block_n_size, head_size])
            
            # (tile_n, HEAD_SIZE)
            k = hl.zeros([block_n_size, head_size], dtype=k_load.dtype)
            # k = hl.full([block_n_size, head_size],float("-inf"), dtype=k_load.dtype)
            tile_token_offsets = hl.arange(block_n_size)
            absolute_tile_token_offsets = tile_n.begin * block_n_size + tile_token_offsets
            k = torch.where(absolute_tile_token_offsets[:, None] < seq_len, 
                            k_load, k)
            # (HEAD_SIZE, tile_n)
            k = k.transpose(0, 1)
            # (tile_n, HEAD_SIZE)
            v = hl.zeros([block_n_size, head_size], dtype=v_load.dtype)
            # v = hl.full([block_n_size, head_size],float("-inf"), dtype=v_load.dtype)
            v = torch.where(absolute_tile_token_offsets[:, None] < seq_len, 
                            v_load, v)
            # (tile_m, tile_n)
            S = hl.dot(q, k, out_dtype=torch.float32) * scale
            # DEBUG: to check the shape...
            # S = S.view([block_m_size, block_n_size])
            # all query heads for one query token are valid
            # block_m_query_mask = tile_q.begin + hl.arange(block_m_size) // num_queries_per_kv
            # block_m_query_mask = tile_q.index.repeat_interleave(tile_m.block_size, dim=0, output_size=block_m_size)
            # block_m_query_mask = tile_q.index.repeat_interleave(tile_m.block_size, dim=0, output_size=S.size(0))
            # block_m_query_mask = tile_m.index // num_queries_per_kv
            # block_m_query_mask = seq_tile.index.repeat_interleave(tile_m.block_size, dim=0)
            # block_m_query_mask = seq_tile.index.repeat_interleave(tile_m.block_size, dim=0) - seq_idx
            # # TODO: not sure we need this for decode only...?
            # block_m_query_mask = hl.arange(block_m_size) // num_queries_per_kv
            # # DEBUG: to check the shape...
            # # block_m_query_mask = block_m_query_mask.view([block_m_size])
            # # construct 2d causal mask
            # causal_mask = absolute_tile_token_offsets[None, :] < seq_len + block_m_query_mask[:, None]
            # S = torch.where(causal_mask, S, float("-inf"))
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
        # t_output[seq_tile, tile_m, :] = acc.view([seq_tile.block_size, tile_m.block_size, head_size])
        t_output[seq_idx, tile_m, :] = acc.view([tile_m.block_size, head_size])



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
    # max_query_len_int: int,
    num_seqs: int,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
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

    # we need to declare "max_query_len" as hl.constexpr, in order to make 
    #  the launch grid correct all the time (i.e. trigger recompilation 
    #  for decode only batches, etc.). At the same time, we don't want to
    #  recompile all the time, hence we do it in steps of powers of 2.
    # max_used_querylen_padded = 1 if max_seqlen_q == 1 \
    #   else torch._inductor.runtime.runtime_utils.next_power_of_2(max_seqlen_q)

    out.fill_(42)
    # print(f"max_seqlen_q: {max_seqlen_q}, num_seqs: {num_seqs}, max_seqlen_k: {max_seqlen_k}")
    # print(cu_seqlens_q)
    # print(seqused_k)
    # print(block_table)
    # print(f"block_table shape: {block_table.shape}, query shape: {q.shape}")
    # print(f"k stride: {k.stride()}, k shape: {k.shape}")

    if max_seqlen_q != 1:
        kernel_helion_v4_attention(
            t_output=out,
            t_query=q,
            t_key_cache=k,
            t_value_cache=v,
            t_block_tables=block_table,
            t_seq_lens=seqused_k,
            scale=softmax_scale,
            # k_scale=k_descale,
            # v_scale=v_descale,
            t_query_start_lens=cu_seqlens_q,
            max_query_len=max_seqlen_q,  # need not to be a tensor
            # max_used_query_len_padded = int(max_used_querylen_padded),
            max_seq_len=max_seqlen_k,
            num_seqs=num_seqs,
            # is_decode_only = max_seqlen_q == 1
        )
    else:
        kernel_helion_v4_attention_decode_only(
            t_output=out,
            t_query=q,
            t_key_cache=k,
            t_value_cache=v,
            t_block_tables=block_table,
            t_seq_lens=seqused_k,
            scale=softmax_scale,
            # k_scale=k_descale,
            # v_scale=v_descale,
            t_query_start_lens=cu_seqlens_q,
            max_query_len=max_seqlen_q,  # need not to be a tensor
            # max_used_query_len_padded = int(max_used_querylen_padded),
            max_seq_len=max_seqlen_k,
            num_seqs=num_seqs,
            # is_decode_only = max_seqlen_q == 1
        )

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
    # max_used_query_len_padded,
    num_seqs,
    is_decode_only,
):
    max_seqlen = t_seq_lens.max()
    # max_query_len = t_query_start_lens.diff().max()
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

# config = nv_config if torch.version.cuda else amd_config

dbg_config = helion.Config(block_sizes=[2, 4], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', 'last', 'last', 'first', 'first', 'last', 'first', '', 'last'], loop_orders=[[1, 2, 0]], num_stages=2, num_warps=4, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, True], range_num_stages=[0, 2], range_unroll_factors=[0, 1], range_warp_specializes=[])
# TODO: bug, it does not use tl.program_id(1) if changing to xyz...? 
# # dbg_config = helion.Config(block_sizes=[2, 4], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', 'last', 'last', 'first', 'first', 'last', 'first', '', 'last'], loop_orders=[[1, 2, 0]], num_stages=2, num_warps=4, pid_type='xyz', range_flattens=[None, False], range_multi_buffers=[None, True], range_num_stages=[0, 2], range_unroll_factors=[0, 1], range_warp_specializes=[])
# dbg_config = helion.Config(block_sizes=[4, 2], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', 'first', 'last', 'first', 'first', 'last', 'first', '', 'last'], loop_orders=[[1, 2, 0]], num_stages=1, num_warps=4, pid_type='persistent_interleaved', range_flattens=[True, False], range_multi_buffers=[True, True], range_num_stages=[1, 2], range_unroll_factors=[0, 1], range_warp_specializes=[])
# dbg_config = helion.Config(block_sizes=[1], indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[64], load_eviction_policies=['last', 'last', 'first', 'last', 'first', '', '', '', 'last', 'last', ''], loop_orders=[[2, 1, 0]], num_stages=1, num_warps=1, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 2], range_unroll_factors=[0, 0], range_warp_specializes=[])
dbg_config = helion.Config(block_sizes=[2, 8], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], l2_groupings=[32], load_eviction_policies=['last', 'last', 'last', 'first', 'first', 'last', 'first', '', 'last'], loop_orders=[[2, 1, 0]], num_stages=2, num_warps=4, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, True], range_num_stages=[0, 2], range_unroll_factors=[0, 1], range_warp_specializes=[])


@helion.kernel(
    allow_warp_specialize=True,
    # static_shapes=False,
    static_shapes=True,
    # dot_precision='ieee',
    # config=config,
    config=dbg_config,
    # configs=nv_configs,
    autotune_baseline_fn=_triton_baseline_fn,
    # autotune_accuracy_check=False,  # since we can't adjust ATOL manually
    autotune_effort="quick",
    # autotune_ignore_errors=True,
    print_output_code=False,
    print_repro=False,
    # debug_dtype_asserts=True,
    index_dtype=torch.int64,
)
def kernel_helion_v3_attention(
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
    # max_used_query_len_padded: hl.constexpr,
    num_seqs,  # must be on cpu
    # to trigger re-compilation for decode only
    is_decode_only: hl.constexpr,
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    q_block_size = hl.register_block_size(1, int(max_query_len))
    # q_block_size = 1
    # q_block_size = hl.register_block_size(1, int(max_used_query_len_padded))
    num_pages_at_once = hl.register_block_size(1, 512//page_size)
    # num_pages_at_once = hl.register_block_size(1, 1)

    for seq_tile, tile_m, tile_q in hl.tile(
        [num_seqs, num_query_heads, max_query_len],
        # [num_seqs, num_query_heads, max_used_query_len_padded],
        block_size=[1, num_queries_per_kv, q_block_size],
    ):
        seq_idx = seq_tile.begin # is scalar
        seq_len = t_seq_lens[seq_idx]
        # TODO: return if seq_len == 0? How does it work with cudagraphs? 
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        # query_len = query_end - query_start
        # context_len = seq_len - query_len

        block_m_size = tile_m.block_size * tile_q.block_size
        block_n_size = num_pages_at_once * page_size
        kv_head_idx = tile_m.begin // num_queries_per_kv
        # kv_head_idx = torch.amin(tile_m.index // num_queries_per_kv)

        adjusted_tile_q_index = query_start + tile_q.index
        # (tile_q, tile_m, HEAD_SIZE)
        q = hl.load(t_query, 
                    # [tile_q.index, tile_m.index, hl.arange(head_size)], 
                    [adjusted_tile_q_index, tile_m.index, hl.arange(head_size)], 
                    # extra_mask=tile_q.index[:, None, None] < query_end,
                    # extra_mask=tile_q.index < query_len,
                    extra_mask=adjusted_tile_q_index[:, None, None] < query_end,
                    # TODO: others??
                    )
        # (tile_m, HEAD_SIZE)
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
            blk_idxs = t_block_tables[seq_idx, tile_n]
            # blk_idxs = hl.load(t_block_tables,
            #     [seq_idx, tile_n],
            #     # TODO: necessary? or how is it with partial tile_n?
            #     extra_mask=tile_n.index[None, :] < num_blocks,
            # )
            blk_idxs = blk_idxs.view([num_pages_at_once])
            # (tile_n, PAGE_SIZE, 1, HEAD_SIZE)
            # k = t_key_cache[blk_idxs, :, kv_head_idx, :]
            tile_offsets = tile_n.begin * page_size + hl.arange(page_size)
            k = hl.load(t_key_cache, 
                       [blk_idxs, hl.arange(page_size), kv_head_idx, hl.arange(head_size)],
                       # mask has only 3 dims, since kv_head_idx is size 1 dim and is removed 
                       extra_mask=tile_offsets[None, :, None] < seq_len
                       )
            # DEBUG: to assert shape
            # k = k.view([tile_n, page_size, head_size])
            # (tile_n, PAGE_SIZE, HEAD_SIZE)
            # v = t_value_cache[blk_idxs, :, kv_head_idx, :]
            v = hl.load(t_value_cache, 
                       [blk_idxs, hl.arange(page_size), kv_head_idx, hl.arange(head_size)],
                       extra_mask=tile_offsets[None, :, None] < seq_len
                       )
            # (HEAD_SIZE, tile_n)
            # TODO: for now, we assume always full pages
            k = k.view([block_n_size, head_size]).transpose(0, 1)
            # (tile_m, tile_n)
            # TODO: why zeros are needed as acc? only in Triton code? 
            #   for future sliding window?
            # S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
            # S = (hl.dot(q, k, out_dtype=torch.float32, acc=S) * scale)
            S = hl.dot(q, k, out_dtype=torch.float32) * scale
            # DEBUG: to check the shape...
            # S = S.view([block_m_size, block_n_size])
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

            # (tile_n, HEAD_SIZE)
            v_view = v.view([block_n_size, head_size])
            # (tile_m, HEAD_SIZE)
            acc = hl.dot(P.to(v.dtype), v_view, out_dtype=torch.float32, acc=acc)

        # epilogue
        acc = acc / L[:, None]
        hl.store(t_output, 
                 # [tile_q.index, tile_m.index, hl.arange(head_size)], 
                 [adjusted_tile_q_index, tile_m.index, hl.arange(head_size)], 
                 # acc.view([-1, tile_m.block_size, head_size]),
                 acc.view([tile_q.block_size, tile_m.block_size, head_size]),
                 # extra_mask=tile_q.index[:, None, None] < query_end
                 # extra_mask=tile_q.index < query_len,
                 extra_mask=adjusted_tile_q_index[:, None, None] < query_end
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
    print(f"max_seqlen_q: {max_seqlen_q}, num_seqs: {num_seqs}, max_seqlen_k: {max_seqlen_k}")
    print(cu_seqlens_q)
    print(seqused_k)
    print(block_table)

    kernel_helion_v3_attention(
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
        num_seqs=num_seqs,
        is_decode_only = max_seqlen_q == 1
    )

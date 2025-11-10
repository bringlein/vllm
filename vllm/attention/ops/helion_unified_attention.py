

import torch

import helion
import helion.language as hl


@helion.kernel(
    config=helion.Config(
        # block_sizes=[32, 2], 
        # block_sizes=[32, 1], 
        block_sizes=[16, 1], 
        indexing='pointer', 
        l2_groupings=[1], num_stages=1, num_warps=8, pid_type='xyz',
    ), 
    static_shapes=True,
    allow_warp_specialize=True,
    # dot_precision='ieee',
    )
def kernel_helion_v2_attention(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    # k_scale,
    # v_scale,
    t_query_start_lens, # [num_seqs+1]
    t_query_slots,
    # num_seqs,
    # unused, to trigger autotuning...?
    max_seqlen,
    max_query_len,
    # is_decode_only: hl.constexpr,
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    num_tokens = t_query.shape[0]
    num_seqs = t_seq_lens.shape[0]

    for seq_idx, kv_head, tile_q in hl.tile([num_seqs, num_kv_heads, max_query_len], block_size=[1, 1, None]):
        kv_head_idx = kv_head.index
        num_par_seq = seq_idx.block_size

        # for now?
        # assert num_par_seq == 1

        seq_len = t_seq_lens[seq_idx]
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len

        # calculating q block index
        # q_block_idxs = torch.where(tile_q.index < query_len, tile_q.index, None)
        # if tile_q.begin > query_len:
        #     continue
        # query_len_in_this_iteration = query_len - tile_q.begin

        for tile_m in hl.tile(kv_head_idx * num_queries_per_kv, (kv_head_idx+1)*num_queries_per_kv, 
                          block_size=num_queries_per_kv):
            block_m_size = tile_m.block_size * tile_q.block_size
            # (tile_q, tile_m, HEAD_SIZE)
            q = t_query[tile_q, tile_m, :]
            
            # TODO: mask q block
            q_padding = hl.zeros([1, tile_m, head_size])  # needs to be creaded with hl. and before torch.where
            q_condition = (t_query_slots[tile_q] == seq_idx.id)[:, None, None]
            # q = torch.where((tile_q.index < query_len).unsqueeze(1).unsqueeze(2), 
            #                 q, 
            #                 q_padding)
            q = torch.where(q_condition,
                            q, 
                            q_padding)
            # q = torch.index_select(q, 0, tile_q.index[:query_len_in_this_iteration])

            # q_repeated = q.expand([seq_idx, tile_q])
            # q_masked = torch.where()
            # q_rel_indeces = t_query_slots[tile_q]
            # q = torch.vstack(torch.tensor_split(q, q_rel_indeces, dim=0))
            
            # (tile_m, HEAD_SIZE)
            # q = q.view([block_m_size, head_size])
            q = q.view([1, block_m_size, head_size])
            # q = q.view([num_par_seq, block_m_size, head_size])

            m = hl.full([num_par_seq, block_m_size], float("-inf"), dtype=torch.float32) # device=q.device)
            # l = hl.full_like(m, 1.0)
            l = hl.full([num_par_seq, block_m_size], 1.0, dtype=torch.float32)
            # (tile_m, HEAD_SIZE)
            acc = hl.zeros([num_par_seq, block_m_size, head_size], dtype=torch.float32)  # , device=q.device)
            
            # num_blocks = (seq_len + page_size - 1) // page_size 
            # adjust for causal mask
            max_seq_prefix_len = context_len + tile_q.end + (tile_m.block_size - 1) // num_queries_per_kv + 1
            max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
            num_blocks = torch.ceil(max_seq_prefix_len / page_size)
            for tile_n in hl.tile(num_blocks, block_size=None):
                block_n_size = tile_n.block_size * page_size
                blk_idxs = t_block_tables[seq_idx, tile_n].view([tile_n.block_size * num_par_seq]) #.view(-1)
                # (tile_n, PAGE_SIZE, 1, HEAD_SIZE)
                # k = t_key_cache[blk_idxs, :, kv_head, :] # .squeeze(2)
                k = t_key_cache[blk_idxs, 0:page_size, kv_head, 0:head_size]
                # k = k.view([tile_n, page_size, head_size])
                # k = k.view([tile_n.block_size * seq_idx.block_size * kv_head.block_size, page_size, head_size])
                k = k.view([num_par_seq, tile_n.block_size * kv_head.block_size, page_size, head_size])
                # (tile_n, PAGE_SIZE, HEAD_SIZE)
                v = t_value_cache[blk_idxs, :, kv_head, :]
                # (HEAD_SIZE, tile_n)
                # k = k.view([block_n_size, head_size]).transpose(0, 1)
                k = k.view([num_par_seq, tile_n.block_size * kv_head.block_size * page_size, head_size]).transpose(1, 2)
                # (tile_m, tile_n)
                # qk = torch.mm(q, k) * scale
                qk = torch.bmm(q.expand([num_par_seq, block_m_size, head_size]), k) * scale
                # qk = q @ k.permute(2, 0, 1)
                # qk = q @ k
                # qk *= scale
                # to check the shape...
                # qk = qk.view([block_m_size, block_n_size])
                # (tile_m)
                # m_j = torch.maximum(m, torch.amax(qk, 1))
                m_j = torch.maximum(m, torch.amax(qk, -1))
                # (tile_m, tile_n)
                # p = torch.exp(qk - m_j[:, None])
                p = torch.exp(qk - m_j[:, :, None])
                # (tile_m, )
                # l_j = torch.sum(p, 1)
                l_j = torch.sum(p, -1)
                # (tile_m, )
                alpha = torch.exp(m - m_j)
                # (tile_m, HEAD_SIZE)
                # acc *= alpha[:, None]
                acc *= alpha[:, :, None]
                l *= alpha + l_j
                m = m_j

                # (tile_n, HEAD_SIZE)
                # v_view = v.view([tile_n.block_size * page_size, head_size])
                v_view = v.view([num_par_seq, tile_n.block_size * page_size * kv_head.block_size, head_size])
                # (tile_m, HEAD_SIZE)
                # acc += torch.mm(p.to(v.dtype), v_view)
                acc += torch.bmm(p.to(v.dtype), v_view)

            # epilogue
            # acc = acc / l[:, None]
            acc = acc / l[:, :, None]
            # t_output[tile_q, tile_m, :] = acc.view([tile_q.block_size, tile_m.block_size, head_size])
            t_output[tile_q, tile_m, :] = acc.view([num_par_seq * tile_q.block_size, tile_m.block_size, head_size])




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
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    # is_decode_only=False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert (
        q.element_size() >= 2 or block_size >= 32
    ), "Block size must be at least 32 for fp8"

    use_alibi_slopes = alibi_slopes is not None

    # num_seqs = len(seqused_k)
    # num_query_heads = q.shape[1]
    # num_kv_heads = k.shape[2]
    # num_queries_per_kv = num_query_heads // num_kv_heads
    # head_size = q.shape[2]
    t_query_slots = torch.empty([q.shape[0]], dtype=seqused_k.dtype)
    for si in range(0, seqused_k.shape[0]):
        t_query_slots[cu_seqlens_q[si]:cu_seqlens_q[si+1]] = si
    # print(t_query_slots)
    # print(k.shape)

    kernel_helion_v2_attention(
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
        t_query_slots=t_query_slots,
        # num_seqs=num_seqs,
        max_seqlen=max_seqlen_k,
        max_query_len=max_seqlen_q,
        # is_decode_only=bool(is_decode_only),
    )



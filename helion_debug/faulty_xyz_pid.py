from __future__ import annotations

import helion.language as hl
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

@triton.jit
def _helion_kernel_helion_v5_attention(t_key_cache, t_seq_lens, t_query_start_lens, t_query, t_block_tables, t_value_cache, t_output, t_key_cache_size_3, t_block_tables_stride_0, t_block_tables_stride_1, t_key_cache_stride_0, t_key_cache_stride_1, t_key_cache_stride_2, t_key_cache_stride_3, t_output_stride_0, t_output_stride_1, t_output_stride_2, t_query_stride_0, t_query_stride_1, t_query_stride_2, t_query_start_lens_stride_0, t_seq_lens_stride_0, t_value_cache_stride_0, t_value_cache_stride_1, t_value_cache_stride_2, t_value_cache_stride_3, num_seqs, scale, _BLOCK_SIZE_3: tl.constexpr, _RDIM_SIZE_4: tl.constexpr, _RDIM_SIZE_5: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr, _SHAPE_DIM: tl.constexpr, _SHAPE_DIM_1: tl.constexpr, _SHAPE_DIM_2: tl.constexpr, _SHAPE_DIM_3: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, mul_1: tl.constexpr, _SHAPE_DIM_10: tl.constexpr, _SHAPE_DIM_11: tl.constexpr, _SHAPE_DIM_18: tl.constexpr, _SHAPE_DIM_19: tl.constexpr, _SHAPE_DIM_20: tl.constexpr, _SHAPE_DIM_21: tl.constexpr, _SHAPE_DIM_22: tl.constexpr):
    # src[helion_unified_attention.py:122]: for seq_tile, tile_m in hl.tile(
    # src[helion_unified_attention.py:123]:     [num_seqs, num_query_heads],
    # src[helion_unified_attention.py:124]:     block_size=[1, num_queries_per_kv],
    # src[helion_unified_attention.py:122-125]: ...
    num_pid_m = tl.cdiv(32, _BLOCK_SIZE_3)
    num_pid_n = num_seqs
    inner_2d_pid = tl.program_id(0)
    num_pid_in_group = 4 * num_pid_n
    group_id = inner_2d_pid // num_pid_in_group
    first_pid_m = group_id * 4
    group_size_m = min(num_pid_m - first_pid_m, 4)
    pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
    pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
    offset_3 = pid_0 * _BLOCK_SIZE_3
    indices_3 = (offset_3 + tl.arange(0, _BLOCK_SIZE_3)).to(tl.int64)
    offset_2 = pid_1
    indices_11 = tl.arange(0, _RDIM_SIZE_4).to(tl.int64)
    indices_12 = tl.arange(0, _RDIM_SIZE_5).to(tl.int64)
    mask_5 = indices_12 < t_key_cache_size_3
    # src[helion_unified_attention.py:127]: seq_len = t_seq_lens[seq_idx]
    seq_len = tl.load(t_seq_lens + offset_2 * t_seq_lens_stride_0, None)
    # src[helion_unified_attention.py:129]: query_start = t_query_start_lens[seq_idx]
    query_start = tl.load(t_query_start_lens + offset_2 * t_query_start_lens_stride_0, None, eviction_policy='evict_last')
    # src[helion_unified_attention.py:130]: query_end = t_query_start_lens[seq_idx + 1]
    add = 1 + offset_2
    query_end = tl.load(t_query_start_lens + add * t_query_start_lens_stride_0, None, eviction_policy='evict_first')
    # src[helion_unified_attention.py:131]: query_len = query_end - query_start
    v_0 = query_end - query_start
    # src[helion_unified_attention.py:132]: context_len = seq_len - query_len
    v_1 = seq_len - v_0
    # src[helion_unified_attention.py:178]: k_load = t_key_cache[blk_idxs, :, kv_head_idx, :]
    symnode_0 = triton_helpers.div_floor_integer(offset_3, 4)
    # src[helion_unified_attention.py:134]: for tile_q in hl.tile(query_len, block_size=q_block_size):
    # src[helion_unified_attention.py:135]:     # block_m_size = tile_m.block_size * tile_q.block_size
    # src[helion_unified_attention.py:136]:     block_m_size = num_queries_per_kv * q_block_size
    # src[helion_unified_attention.py:134-254]: ...
    for offset_9 in tl.range(0, v_0.to(tl.int64), _BLOCK_SIZE_0, loop_unroll_factor=1, num_stages=2, disallow_acc_multi_buffer=False, flatten=False):
        indices_9 = offset_9 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int64)
        mask_0 = indices_9 < v_0
        query_start_copy = query_start
        query_end_copy = query_end
        v_1_copy = v_1
        seq_len_copy = seq_len
        query_start_copy_0 = query_start_copy
        query_end_copy_0 = query_end_copy
        v_1_copy_0 = v_1_copy
        seq_len_copy_0 = seq_len_copy
        # src[helion_unified_attention.py:136]: block_m_size = num_queries_per_kv * q_block_size
        mul = 4 * _BLOCK_SIZE_0
        # src[helion_unified_attention.py:142]: adjusted_tile_q_index = query_start + tile_q.begin + hl.arange(q_block_size)
        v_2 = tl.cast(query_start_copy_0, tl.int64)
        v_3 = v_2 + offset_9
        iota = tl.arange(0, _BLOCK_SIZE_0).to(tl.int64)
        v_4 = tl.cast(v_3, tl.int64)
        v_5 = v_4 + iota
        # src[helion_unified_attention.py:143]: query_head_offset = tile_m.begin + hl.arange(num_queries_per_kv)
        iota_1 = tl.arange(0, 4).to(tl.int64)
        v_6 = iota_1 + offset_3
        # src[helion_unified_attention.py:144]: q_load_mask = adjusted_tile_q_index[:, None, None] < query_end
        subscript = v_5[:, None, None]
        v_7 = tl.cast(query_end_copy_0, tl.int64)
        v_8 = subscript < v_7
        # src[helion_unified_attention.py:148]: [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
        iota_2 = tl.arange(0, 128).to(tl.int64)
        # src[helion_unified_attention.py:146]: q = hl.load(
        # src[helion_unified_attention.py:147]:     t_query,
        # src[helion_unified_attention.py:148]:     [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
        # src[helion_unified_attention.py:146-151]: ...
        q = tl.load(t_query + (v_5[:, None, None] * t_query_stride_0 + v_6[None, :, None] * t_query_stride_1 + iota_2[None, None, :] * t_query_stride_2), mask_0[:, None, None] & v_8, other=0, eviction_policy='evict_first')
        # src[helion_unified_attention.py:153]: q = q.flatten(start_dim=0, end_dim=1)
        q_1 = tl.reshape(q, [_SHAPE_DIM, 128])
        # src[helion_unified_attention.py:157]: M = hl.full([block_m_size], float("-inf"), dtype=torch.float32)
        M = tl.full([_SHAPE_DIM_1], float('-inf'), tl.float32)
        # src[helion_unified_attention.py:158]: L = hl.full([block_m_size], 1.0, dtype=torch.float32)
        L = tl.full([_SHAPE_DIM_2], 1.0, tl.float32)
        # src[helion_unified_attention.py:159]: acc = hl.zeros([block_m_size, head_size], dtype=torch.float32)
        acc = tl.full([_SHAPE_DIM_3, 128], 0.0, tl.float32)
        # src[helion_unified_attention.py:162]: max_seq_prefix_len = context_len + tile_q.begin + block_m_size + 1
        v_9 = tl.cast(v_1_copy_0, tl.int64)
        v_10 = v_9 + offset_9
        v_11 = tl.cast(v_10, tl.int64)
        v_12 = v_11 + mul
        v_13 = tl.full([], 1, tl.int32)
        v_14 = v_12 + v_13
        # src[helion_unified_attention.py:163]: max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
        v_15 = triton_helpers.minimum(v_14, seq_len_copy_0)
        # src[helion_unified_attention.py:164]: num_blocks = torch.ceil(max_seq_prefix_len / page_size)
        v_16 = tl.cast(v_15, tl.float32)
        v_17 = 0.0625
        v_18 = v_16 * v_17
        v_19 = libdevice.ceil(v_18)
        # src[helion_unified_attention.py:165]: for tile_n in hl.tile(num_blocks, block_size=num_pages_at_once):
        # src[helion_unified_attention.py:166]:     block_n_size = num_pages_at_once * page_size
        # src[helion_unified_attention.py:167]:     # TODO: bug: will not be right shape if tile_n is partial
        # src[helion_unified_attention.py:165-245]: ...
        for offset_10 in tl.range(0, v_19.to(tl.int64), _BLOCK_SIZE_1, loop_unroll_factor=1, num_stages=1, disallow_acc_multi_buffer=True):
            indices_10 = offset_10 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int64)
            mask_1 = indices_10 < v_19
            seq_len_copy_0_copy = seq_len_copy_0
            q_1_copy = q_1
            v_1_copy_0_copy = v_1_copy_0
            M_copy = M
            acc_copy = acc
            L_copy = L
            seq_len_copy_0_copy_0 = seq_len_copy_0_copy
            q_1_copy_0 = q_1_copy
            v_1_copy_0_copy_0 = v_1_copy_0_copy
            M_copy_0 = M_copy
            acc_copy_0 = acc_copy
            L_copy_0 = L_copy
            # src[helion_unified_attention.py:171]: [seq_idx, tile_n.begin + hl.arange(num_pages_at_once)],
            iota_3 = tl.arange(0, _BLOCK_SIZE_1).to(tl.int64)
            v_20 = iota_3 + offset_10
            # src[helion_unified_attention.py:169]: blk_idxs = hl.load(
            # src[helion_unified_attention.py:170]:     t_block_tables,
            # src[helion_unified_attention.py:171]:     [seq_idx, tile_n.begin + hl.arange(num_pages_at_once)],
            # src[helion_unified_attention.py:169-174]: ...
            blk_idxs = tl.load(t_block_tables + (offset_2 * t_block_tables_stride_0 + v_20 * t_block_tables_stride_1), mask_1, other=0)
            # src[helion_unified_attention.py:175]: blk_idxs = blk_idxs.view([num_pages_at_once]).to(torch.int64)
            view = tl.reshape(blk_idxs, [_BLOCK_SIZE_1])
            v_21 = tl.cast(view, tl.int64)
            # src[helion_unified_attention.py:178]: k_load = t_key_cache[blk_idxs, :, kv_head_idx, :]
            k_load = tl.load(t_key_cache + (v_21[:, None, None] * t_key_cache_stride_0 + indices_11[None, :, None] * t_key_cache_stride_1 + symnode_0 * t_key_cache_stride_2 + indices_12[None, None, :] * t_key_cache_stride_3), mask_1[:, None, None] & mask_5[None, None, :], other=0, eviction_policy='evict_last')
            # src[helion_unified_attention.py:181]: k_load = k_load.flatten(start_dim=0, end_dim=1)
            k_load_1 = tl.reshape(k_load, [_SHAPE_DIM_10, 128])
            # src[helion_unified_attention.py:185]: k = hl.zeros([block_n_size, head_size], dtype=k_load.dtype)
            k = tl.full([_SHAPE_DIM_11, 128], 0.0, tl.bfloat16)
            # src[helion_unified_attention.py:186]: absolute_tile_token_offsets = tile_n.begin * page_size + hl.arange(
            mul_4 = 16 * offset_10
            # src[helion_unified_attention.py:186]: absolute_tile_token_offsets = tile_n.begin * page_size + hl.arange(
            # src[helion_unified_attention.py:187]:     block_n_size
            # src[helion_unified_attention.py:188]: )
            iota_4 = tl.arange(0, mul_1).to(tl.int64)
            v_22 = iota_4 + mul_4
            # src[helion_unified_attention.py:190]: absolute_tile_token_offsets[:, None] < seq_len, k_load, k
            subscript_1 = v_22[:, None]
            v_23 = tl.cast(seq_len_copy_0_copy_0, tl.int64)
            v_24 = subscript_1 < v_23
            # src[helion_unified_attention.py:189]: k = torch.where(
            # src[helion_unified_attention.py:190]:     absolute_tile_token_offsets[:, None] < seq_len, k_load, k
            # src[helion_unified_attention.py:191]: )
            v_25 = tl.where(v_24, k_load_1, k)
            # src[helion_unified_attention.py:193]: k = k.transpose(0, 1)
            k_2 = tl.permute(v_25, [1, 0])
            # src[helion_unified_attention.py:196]: v_load = t_value_cache[blk_idxs, :, kv_head_idx, :]
            v_load = tl.load(t_value_cache + (v_21[:, None, None] * t_value_cache_stride_0 + indices_11[None, :, None] * t_value_cache_stride_1 + symnode_0 * t_value_cache_stride_2 + indices_12[None, None, :] * t_value_cache_stride_3), mask_1[:, None, None] & mask_5[None, None, :], other=0, eviction_policy='evict_last')
            # src[helion_unified_attention.py:197]: v_load = v_load.flatten(start_dim=0, end_dim=1)
            v_load_1 = tl.reshape(v_load, [_SHAPE_DIM_18, 128])
            # src[helion_unified_attention.py:201]: v = hl.zeros([block_n_size, head_size], dtype=v_load.dtype)
            v = tl.full([_SHAPE_DIM_19, 128], 0.0, tl.bfloat16)
            # src[helion_unified_attention.py:203]: absolute_tile_token_offsets[:, None] < seq_len, v_load, v
            subscript_2 = v_22[:, None]
            v_26 = tl.cast(seq_len_copy_0_copy_0, tl.int64)
            v_27 = subscript_2 < v_26
            # src[helion_unified_attention.py:202]: v = torch.where(
            # src[helion_unified_attention.py:203]:     absolute_tile_token_offsets[:, None] < seq_len, v_load, v
            # src[helion_unified_attention.py:204]: )
            v_28 = tl.where(v_27, v_load_1, v)
            # src[helion_unified_attention.py:209]: S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
            S = tl.full([_SHAPE_DIM_20, _SHAPE_DIM_21], 0.0, tl.float32)
            # src[helion_unified_attention.py:210]: S = hl.dot(q, k, out_dtype=torch.float32, acc=S) * scale
            dot = tl.dot(tl.cast(q_1_copy_0, tl.bfloat16), tl.cast(k_2, tl.bfloat16), acc=S, input_precision='tf32', out_dtype=tl.float32)
            v_29 = dot * scale
            # src[helion_unified_attention.py:219]: block_m_query_mask = tile_q.begin + hl.arange(
            # src[helion_unified_attention.py:220]:     q_block_size
            # src[helion_unified_attention.py:221]: ).repeat_interleave(num_queries_per_kv, dim=0)
            iota_5 = tl.arange(0, _BLOCK_SIZE_0).to(tl.int64)
            unsqueeze = iota_5[:, None]
            expand = tl.broadcast_to(unsqueeze, [_BLOCK_SIZE_0, 4])
            view_3 = tl.reshape(expand, [_SHAPE_DIM_22])
            v_30 = view_3 + offset_9
            # src[helion_unified_attention.py:226]: absolute_tile_token_offsets[None, :]
            subscript_3 = v_22[None, :]
            # src[helion_unified_attention.py:227]: < context_len + block_m_query_mask[:, None] + 1
            subscript_4 = v_30[:, None]
            v_31 = tl.cast(v_1_copy_0_copy_0, tl.int64)
            v_32 = v_31 + subscript_4
            v_33 = tl.full([], 1, tl.int64)
            v_34 = v_32 + v_33
            # src[helion_unified_attention.py:226]: absolute_tile_token_offsets[None, :]
            # src[helion_unified_attention.py:227]: < context_len + block_m_query_mask[:, None] + 1
            v_35 = subscript_3 < v_34
            # src[helion_unified_attention.py:229]: S = torch.where(causal_mask, S, float("-inf"))
            v_36 = float('-inf')
            v_37 = tl.where(v_35, v_29, v_36)
            # src[helion_unified_attention.py:232]: M_j = torch.maximum(M, torch.amax(S, 1))
            amax = tl.cast(tl.max(v_37, 1), tl.float32)
            v_38 = triton_helpers.maximum(M_copy_0, amax)
            # src[helion_unified_attention.py:234]: P = torch.exp(S - M_j[:, None])
            subscript_5 = v_38[:, None]
            v_39 = v_37 - subscript_5
            v_40 = libdevice.exp(v_39)
            # src[helion_unified_attention.py:236]: L_j = torch.sum(P, 1)
            L_j = tl.cast(tl.sum(v_40, 1), tl.float32)
            # src[helion_unified_attention.py:238]: alpha = torch.exp(M - M_j)
            v_41 = M_copy_0 - v_38
            v_42 = libdevice.exp(v_41)
            # src[helion_unified_attention.py:240]: acc = acc * alpha[:, None]
            subscript_6 = v_42[:, None]
            v_43 = acc_copy_0 * subscript_6
            # src[helion_unified_attention.py:241]: L = (L * alpha) + L_j
            v_44 = L_copy_0 * v_42
            L = v_44 + L_j
            # src[helion_unified_attention.py:242]: M = M_j
            M = v_38
            # src[helion_unified_attention.py:245]: acc = hl.dot(P.to(v.dtype), v, out_dtype=torch.float32, acc=acc)
            v_46 = tl.cast(v_40, tl.bfloat16)
            acc = tl.dot(tl.cast(v_46, tl.bfloat16), tl.cast(v_28, tl.bfloat16), acc=v_43, input_precision='tf32', out_dtype=tl.float32)
        # src[helion_unified_attention.py:248]: acc = acc / L[:, None]
        subscript_7 = L[:, None]
        v_47 = acc / subscript_7
        # src[helion_unified_attention.py:251]: [adjusted_tile_q_index, tile_m.index, hl.arange(head_size)],
        iota_6 = tl.arange(0, 128).to(tl.int64)
        # src[helion_unified_attention.py:252]: acc.view([q_block_size, num_queries_per_kv, head_size]),
        view_1 = tl.reshape(v_47, [_BLOCK_SIZE_0, 4, 128])
        # src[helion_unified_attention.py:249]: hl.store(
        # src[helion_unified_attention.py:250]:     t_output,
        # src[helion_unified_attention.py:251]:     [adjusted_tile_q_index, tile_m.index, hl.arange(head_size)],
        # src[helion_unified_attention.py:249-254]: ...
        v_48 = tl.cast(view_1, tl.bfloat16)
        tl.store(t_output + (v_5[:, None, None] * t_output_stride_0 + indices_3[None, :, None] * t_output_stride_1 + iota_6[None, None, :] * t_output_stride_2), v_48, mask_0[:, None, None] & v_8)

def kernel_helion_v5_attention(t_output, t_query, t_key_cache, t_value_cache, t_block_tables, t_seq_lens, scale, t_query_start_lens, num_seqs, q_block_padded_size: hl.constexpr, *, _launcher=_default_launcher):
    # src[helion_unified_attention.py:110]: head_size = hl.specialize(t_query.size(2))
    head_size = 128
    # src[helion_unified_attention.py:113]: page_size = hl.specialize(t_value_cache.size(1))
    page_size = 16
    # src[helion_unified_attention.py:116]: assert page_size == t_key_cache.size(1)
    assert page_size == t_key_cache.size(1)
    # src[helion_unified_attention.py:117]: assert head_size == t_key_cache.size(3)
    assert head_size == t_key_cache.size(3)
    # src[helion_unified_attention.py:122]: for seq_tile, tile_m in hl.tile(
    # src[helion_unified_attention.py:123]:     [num_seqs, num_query_heads],
    # src[helion_unified_attention.py:124]:     block_size=[1, num_queries_per_kv],
    # src[helion_unified_attention.py:122-125]: ...
    _BLOCK_SIZE_3 = 4
    _RDIM_SIZE_4 = 16
    _RDIM_SIZE_5 = triton.next_power_of_2(t_key_cache.size(3))
    # src[helion_unified_attention.py:134]: for tile_q in hl.tile(query_len, block_size=q_block_size):
    # src[helion_unified_attention.py:135]:     # block_m_size = tile_m.block_size * tile_q.block_size
    # src[helion_unified_attention.py:136]:     block_m_size = num_queries_per_kv * q_block_size
    # src[helion_unified_attention.py:134-254]: ...
    _BLOCK_SIZE_0 = 32
    # src[helion_unified_attention.py:153]: q = q.flatten(start_dim=0, end_dim=1)
    _SHAPE_DIM = 4 * _BLOCK_SIZE_0
    # src[helion_unified_attention.py:157]: M = hl.full([block_m_size], float("-inf"), dtype=torch.float32)
    _SHAPE_DIM_1 = 4 * _BLOCK_SIZE_0
    # src[helion_unified_attention.py:158]: L = hl.full([block_m_size], 1.0, dtype=torch.float32)
    _SHAPE_DIM_2 = 4 * _BLOCK_SIZE_0
    # src[helion_unified_attention.py:159]: acc = hl.zeros([block_m_size, head_size], dtype=torch.float32)
    _SHAPE_DIM_3 = 4 * _BLOCK_SIZE_0
    # src[helion_unified_attention.py:165]: for tile_n in hl.tile(num_blocks, block_size=num_pages_at_once):
    # src[helion_unified_attention.py:166]:     block_n_size = num_pages_at_once * page_size
    # src[helion_unified_attention.py:167]:     # TODO: bug: will not be right shape if tile_n is partial
    # src[helion_unified_attention.py:165-245]: ...
    _BLOCK_SIZE_1 = 2
    # src[helion_unified_attention.py:178]: k_load = t_key_cache[blk_idxs, :, kv_head_idx, :]
    _SHAPE_DIM_4 = t_value_cache.size(1)
    _SHAPE_DIM_5 = t_query.size(2)
    _SHAPE_DIM_6 = t_value_cache.size(1)
    _SHAPE_DIM_7 = t_query.size(2)
    _SHAPE_DIM_8 = t_value_cache.size(1)
    _SHAPE_DIM_9 = t_query.size(2)
    # src[helion_unified_attention.py:181]: k_load = k_load.flatten(start_dim=0, end_dim=1)
    _SHAPE_DIM_10 = 16 * _BLOCK_SIZE_1
    # src[helion_unified_attention.py:185]: k = hl.zeros([block_n_size, head_size], dtype=k_load.dtype)
    _SHAPE_DIM_11 = 16 * _BLOCK_SIZE_1
    # src[helion_unified_attention.py:196]: v_load = t_value_cache[blk_idxs, :, kv_head_idx, :]
    _SHAPE_DIM_12 = t_value_cache.size(1)
    _SHAPE_DIM_13 = t_query.size(2)
    _SHAPE_DIM_14 = t_value_cache.size(1)
    _SHAPE_DIM_15 = t_query.size(2)
    _SHAPE_DIM_16 = t_value_cache.size(1)
    _SHAPE_DIM_17 = t_query.size(2)
    # src[helion_unified_attention.py:197]: v_load = v_load.flatten(start_dim=0, end_dim=1)
    _SHAPE_DIM_18 = 16 * _BLOCK_SIZE_1
    # src[helion_unified_attention.py:201]: v = hl.zeros([block_n_size, head_size], dtype=v_load.dtype)
    _SHAPE_DIM_19 = 16 * _BLOCK_SIZE_1
    # src[helion_unified_attention.py:209]: S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
    _SHAPE_DIM_20 = 4 * _BLOCK_SIZE_0
    _SHAPE_DIM_21 = 16 * _BLOCK_SIZE_1
    # src[helion_unified_attention.py:219]: block_m_query_mask = tile_q.begin + hl.arange(
    # src[helion_unified_attention.py:220]:     q_block_size
    # src[helion_unified_attention.py:221]: ).repeat_interleave(num_queries_per_kv, dim=0)
    _SHAPE_DIM_22 = 4 * _BLOCK_SIZE_0
    # src[helion_unified_attention.py:122]: for seq_tile, tile_m in hl.tile(
    # src[helion_unified_attention.py:123]:     [num_seqs, num_query_heads],
    # src[helion_unified_attention.py:124]:     block_size=[1, num_queries_per_kv],
    # src[helion_unified_attention.py:122-254]: ...
    _RDIM_SIZE_6 = triton.next_power_of_2(16 * _BLOCK_SIZE_1)
    _launcher(_helion_kernel_helion_v5_attention, (triton.cdiv(t_query.size(1), _BLOCK_SIZE_3), num_seqs), t_key_cache, t_seq_lens, t_query_start_lens, t_query, t_block_tables, t_value_cache, t_output, t_key_cache.size(3), t_block_tables.stride(0), t_block_tables.stride(1), t_key_cache.stride(0), t_key_cache.stride(1), t_key_cache.stride(2), t_key_cache.stride(3), t_output.stride(0), t_output.stride(1), t_output.stride(2), t_query.stride(0), t_query.stride(1), t_query.stride(2), t_query_start_lens.stride(0), t_seq_lens.stride(0), t_value_cache.stride(0), t_value_cache.stride(1), t_value_cache.stride(2), t_value_cache.stride(3), num_seqs, scale, _BLOCK_SIZE_3, _RDIM_SIZE_4, _RDIM_SIZE_5, _BLOCK_SIZE_0, _SHAPE_DIM, _SHAPE_DIM_1, _SHAPE_DIM_2, _SHAPE_DIM_3, _BLOCK_SIZE_1, 16 * _BLOCK_SIZE_1, _SHAPE_DIM_10, _SHAPE_DIM_11, _SHAPE_DIM_18, _SHAPE_DIM_19, _SHAPE_DIM_20, _SHAPE_DIM_21, _SHAPE_DIM_22, num_warps=8, num_stages=7)
"""Fused RoPE + layout transform kernel for Qwen-Image MMDiT.

Takes Q/K after QK-norm + pre-computed Q_swap/K_swap, applies RoPE,
and writes output in transposed SDPA layout.

Replaces per-block: 4x ttnn.multiply + 2x ttnn.add + 2x ttnn.concat + 3x ttnn.transpose
= 11 ttnn kernel launches fused into 1 tt-lang kernel.

Input layout:  Q/K are [S, H*D] (2D, where H=heads_per_chip, D=HEAD_DIM)
               cos/sin_perm are [S, H*D] (same shape, broadcast across heads)
Output layout: [H*S, D] (transposed: each head's S-by-D block is contiguous)

Grid="auto" parallelizes over (seq_tile, head) pairs.
"""
import ttl

TILE = 32


def make_rope_layout_kernel(n_heads, head_tiles):
    """Factory: creates a fused RoPE + layout kernel.

    Args:
        n_heads: number of attention heads per chip (e.g. 6 with TP=4)
        head_tiles: tiles per head dimension (HEAD_DIM // TILE, e.g. 4 for 128)
    """
    @ttl.kernel(grid="auto")
    def rope_layout(q, q_swap, k, k_swap, v,
                    cos_tab, sin_tab,
                    q_out, k_out, v_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = q.shape[0] // TILE
        total_units = seq_tiles * n_heads
        units_per_core = -(-total_units // grid_cols)

        q_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)
        qs_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)
        ks_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)
        qr_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)
        kr_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)
        vo_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, head_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_u in range(units_per_core):
                uid = core_x * units_per_core + local_u
                if uid < total_units:
                    with cos_dfb.wait() as cv, sin_dfb.wait() as sv:
                        with q_dfb.wait() as qv, qs_dfb.wait() as qsv, qr_dfb.reserve() as qr:
                            qr.store(qv * cv + qsv * sv)
                        with k_dfb.wait() as kv, ks_dfb.wait() as ksv, kr_dfb.reserve() as kr:
                            kr.store(kv * cv + ksv * sv)
                    with v_dfb.wait() as vv, vo_dfb.reserve() as vo:
                        vo.store(vv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_u in range(units_per_core):
                uid = core_x * units_per_core + local_u
                if uid < total_units:
                    row = uid // n_heads
                    h = uid % n_heads
                    hc = h * head_tiles
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(q[row, hc:hc + head_tiles], blk); tx.wait()
                    with qs_dfb.reserve() as blk:
                        tx = ttl.copy(q_swap[row, hc:hc + head_tiles], blk); tx.wait()
                    with k_dfb.reserve() as blk:
                        tx = ttl.copy(k[row, hc:hc + head_tiles], blk); tx.wait()
                    with ks_dfb.reserve() as blk:
                        tx = ttl.copy(k_swap[row, hc:hc + head_tiles], blk); tx.wait()
                    with v_dfb.reserve() as blk:
                        tx = ttl.copy(v[row, hc:hc + head_tiles], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[row, hc:hc + head_tiles], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[row, hc:hc + head_tiles], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_u in range(units_per_core):
                uid = core_x * units_per_core + local_u
                if uid < total_units:
                    row = uid // n_heads
                    h = uid % n_heads
                    # Output layout: [H*seq_tiles, head_tiles]
                    # Head h, seq row r -> output row h*seq_tiles + r
                    out_row = h * seq_tiles + row
                    with qr_dfb.wait() as blk:
                        tx = ttl.copy(blk, q_out[out_row, 0:head_tiles]); tx.wait()
                    with kr_dfb.wait() as blk:
                        tx = ttl.copy(blk, k_out[out_row, 0:head_tiles]); tx.wait()
                    with vo_dfb.wait() as blk:
                        tx = ttl.copy(blk, v_out[out_row, 0:head_tiles]); tx.wait()

    return rope_layout


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)

    # Test dimensions matching Qwen-Image with TP=4
    SEQ = 256   # sequence length (image tokens for 256x256)
    N_HEADS = 6  # heads per chip
    HEAD_DIM = 128
    D_MODEL = N_HEADS * HEAD_DIM  # 768 per chip
    HEAD_TILES = HEAD_DIM // TILE  # 4

    print(f"Testing RoPE+layout kernel: SEQ={SEQ}, N_HEADS={N_HEADS}, HEAD_DIM={HEAD_DIM}")

    # Create test data
    torch.manual_seed(42)
    q_pt = torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16)
    k_pt = torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16)
    v_pt = torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16)

    # Build swap: adjacent element swap within each head
    # [x0,x1,x2,x3,...] -> [x1,x0,x3,x2,...]
    swap_indices = torch.arange(D_MODEL)
    for h in range(N_HEADS):
        base = h * HEAD_DIM
        for i in range(0, HEAD_DIM, 2):
            swap_indices[base + i] = base + i + 1
            swap_indices[base + i + 1] = base + i
    q_swap_pt = q_pt[:, swap_indices]
    k_swap_pt = k_pt[:, swap_indices]

    # cos/sin tables (same for each head, broadcast)
    cos_raw = torch.randn(SEQ, HEAD_DIM, dtype=torch.bfloat16)
    sin_raw = torch.randn(SEQ, HEAD_DIM, dtype=torch.bfloat16)
    # Apply sign pattern to sin: [-sin, sin, -sin, sin, ...]
    sign = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
    sign[0::2] = -1
    sin_perm_raw = sin_raw * sign.unsqueeze(0)

    # Expand cos/sin to full D_MODEL (repeat across heads)
    cos_pt = cos_raw.repeat(1, N_HEADS)  # [SEQ, D_MODEL]
    sin_perm_pt = sin_perm_raw.repeat(1, N_HEADS)  # [SEQ, D_MODEL]

    # Reference: compute expected output
    q_roped_ref = q_pt * cos_pt + q_swap_pt * sin_perm_pt
    k_roped_ref = k_pt * cos_pt + k_swap_pt * sin_perm_pt

    # Reshape to [SEQ, N_HEADS, HEAD_DIM] then transpose to [N_HEADS*SEQ, HEAD_DIM]
    q_ref_3d = q_roped_ref.view(SEQ, N_HEADS, HEAD_DIM)
    k_ref_3d = k_roped_ref.view(SEQ, N_HEADS, HEAD_DIM)
    v_ref_3d = v_pt.view(SEQ, N_HEADS, HEAD_DIM)
    # Transpose: [S, H, D] -> [H, S, D] -> [H*S, D]
    q_ref_out = q_ref_3d.permute(1, 0, 2).reshape(N_HEADS * SEQ, HEAD_DIM)
    k_ref_out = k_ref_3d.permute(1, 0, 2).reshape(N_HEADS * SEQ, HEAD_DIM)
    v_ref_out = v_ref_3d.permute(1, 0, 2).reshape(N_HEADS * SEQ, HEAD_DIM)

    # Upload to device
    def to_dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    q_tt = to_dev(q_pt)
    q_swap_tt = to_dev(q_swap_pt)
    k_tt = to_dev(k_pt)
    k_swap_tt = to_dev(k_swap_pt)
    v_tt = to_dev(v_pt)
    cos_tt = to_dev(cos_pt)
    sin_tt = to_dev(sin_perm_pt)

    # Allocate outputs
    q_out_tt = to_dev(torch.zeros(N_HEADS * SEQ, HEAD_DIM, dtype=torch.bfloat16))
    k_out_tt = to_dev(torch.zeros(N_HEADS * SEQ, HEAD_DIM, dtype=torch.bfloat16))
    v_out_tt = to_dev(torch.zeros(N_HEADS * SEQ, HEAD_DIM, dtype=torch.bfloat16))

    # Create and run kernel
    rope_kernel = make_rope_layout_kernel(N_HEADS, HEAD_TILES)
    rope_kernel(q_tt, q_swap_tt, k_tt, k_swap_tt, v_tt,
                cos_tt, sin_tt,
                q_out_tt, k_out_tt, v_out_tt)

    # Read back and compare
    q_result = ttnn.to_torch(q_out_tt).float()
    k_result = ttnn.to_torch(k_out_tt).float()
    v_result = ttnn.to_torch(v_out_tt).float()

    def pcc(a, b):
        a, b = a.flatten(), b.flatten()
        a, b = a - a.mean(), b - b.mean()
        return (a * b).sum() / (a.norm() * b.norm() + 1e-8)

    q_pcc = pcc(q_result, q_ref_out.float())
    k_pcc = pcc(k_result, k_ref_out.float())
    v_pcc = pcc(v_result, v_ref_out.float())

    print(f"Q PCC: {q_pcc:.6f} {'PASS' if q_pcc > 0.999 else 'FAIL'}")
    print(f"K PCC: {k_pcc:.6f} {'PASS' if k_pcc > 0.999 else 'FAIL'}")
    print(f"V PCC: {v_pcc:.6f} {'PASS' if v_pcc > 0.999 else 'FAIL'}")

    ttnn.close_device(device)
    print("Done!")

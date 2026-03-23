"""Broadcast row 0 of a tile-padded tensor to all rows of the output.

Takes inp [1_tile_row, D_tiles] and writes to out [S_tiles, D_tiles].
The input is typically [1,D] padded to [32,D], so only row 0 of each tile
has real data. We use ttl.math.broadcast(dims=[0]) to replicate row 0
across all 32 rows within each tile, then write to every output tile row.
"""
import ttl

TILE = 32
ELEM_GRAN = 8


@ttl.kernel(grid="auto")
def broadcast_row_kernel(inp, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    out_row_tiles = out.shape[0] // TILE
    col_blocks = out.shape[1] // TILE // ELEM_GRAN
    total = out_row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, ELEM_GRAN), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with inp_dfb.wait() as iv, bcast_dfb.reserve() as bc:
                    bc.store(ttl.math.broadcast(iv, dims=[0]))
                with bcast_dfb.wait() as bv, out_dfb.reserve() as o:
                    o.store(bv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with inp_dfb.reserve() as blk:
                    tx = ttl.copy(inp[0, sc:sc + ELEM_GRAN], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

// TILE_SIZE is supplied by the host during OpenCL program build.

__kernel void matrix_multiply_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int N
)
{
    // Global position of the output element.
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    // Position of this work-item inside its work-group.
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    // Shared tiles stored in fast local memory.
    __local float tile_A[TILE_SIZE][TILE_SIZE];
    __local float tile_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Process the matrices one tile at a time.
    for (int tile = 0; tile < N; tile += TILE_SIZE) {

        // Cooperatively load one element of A and B.
        tile_A[local_row][local_col] =
            A[row * N + (tile + local_col)];

        tile_B[local_row][local_col] =
            B[(tile + local_row) * N + col];

        // Make sure the entire tile is loaded before using it.
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reuse the data from local memory.
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[local_row][k] *
                   tile_B[k][local_col];
        }

        // Make sure all work-items have finished using
        // the current tile before loading the next one.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row * N + col] = sum;
}

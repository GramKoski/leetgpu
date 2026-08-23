#include <cuda_runtime.h>

#define tile 16

__global__ void tiled_matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // ty is row index in input tile
    // tx is col index in input tile (varies with threadIdx.x)

    __shared__ tile[TILE][TILE];

    int ty = threadIdx.y % TILE;
    int tx = threadIdx.x % TILE;

    // Load tile, every thread loads one element
    tile[ty][tx] = input[(blockIdx.y * TILE + ty) * cols + (blockIdx.x * TILE + tx)]

    // Make sure an entire block is loaded
    __syncthreads();

    // Coallesced HBM write
    // 1. We to tile-wise transpose by swapping blockIdx.y and blockIdx.x
    // 2. We to intra-tile transpose by indexing tile[tx][ty]
    // 3. We don't swap ty and tx on the output because we already are doing transpose implicitly in smem

    output[(blockIdx.x*TILE + ty) * rows + (blockIdx.y*TILE + tx)] = tile[tx][ty];

}

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Naive kernel, every thread owns one index of input. We simply write that index to the transposed output
    if (row < rows && col < cols) {
        output[col*rows + row] = input[row*cols + col];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);

    // We see that there is one thread per element threadsPerBlock * blocksPerGrid = (rows, cols) (with ceiling considered)
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

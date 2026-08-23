#include <cuda_runtime.h>

#define TILE 16


__global__ void tiled_matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Together this thread produces one output element of C at (row, col)
    // This means the thread will iterate across the column index of A and accumulate
    // It will iterate across the row index of B and accumulate
    int row = blockIdx.y * blockDim.y + threadIdx.y;    
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Use shared memory to store tiles of A and B such that we don't have to pull from HBM every iteration
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0;

    for (int t = 0; t < (N+TILE-1)/TILE; ++t) {
        // We want to fetch a single 16 x 16 tile that will be used for this computation
        // One thread will use a single intra-tile row of A and a single intra-tile column of B
        // However, this thread only fetches one element and the teammates do the rest
        // As we move 0...->15 in the thread block column dimension (threadIdx.x varies) we load one corresponding tile-wise row of A (col dim of A varies)
        // As we move 0...->15 in the thread block row dimension (threadIdx.y varies) we load one corresponding tile-wise col of B (row dim of B varies)
        int aCol = t*TILE + threadIdx.x;
        int bRow = t*TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < N) ? A[row*N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < K) ? B[bRow*K + col] : 0.0f;
        // Note that col varies with threadIdx.x--this means that the memory accesses on B are coalesced
        // Note the same thing with aCol

        // Make sure the entire tile is in smem
        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // Make sure faster threads don't go to the next t-loop iteration and overwrite smem while slower threads are still reading.
        __syncthreads();

    }

    if (row < M && col < K) {
        C[row*K + col] = sum;
    }
}

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Remember that there are 16 threads in this row
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0;   // Register-allocated (probably)
        // Responsible for calculating C[row][col]
        for (int n = 0; n < N; ++n) {
            sum += A[row*N + n] * B[n * K + col];   // This index math will take some getting used to
            // Note that the memory is coalesced nicely--we read adjacent blocks of B and just two memory addresses of A per warp per iteration
        }
        C[row*K + col] = sum;   // Global memory right
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    // 16 * 16 = 256
    // 1 <= M, N, K <= 8192
    // 8192 / 16 = 512
    // Each thread handles a single element of the output -- one block handles a 16x16 tile of the output
    // In largest size the grid is 512 x 512 = 262144
    // What I learned is all blocks don't need to be resident on the GPU at once--blocks will be queued "in hard"
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    tiled_matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
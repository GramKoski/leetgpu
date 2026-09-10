#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#define WM 32   // warp tile rows (How many rows of output C does each warp compute)
#define WN 32   // warp tile cols (How many cols of output C does each warp compute)
#define BM 64   // block tile rows
#define BN 64   // block tile cols
#define BK 16   // K-slab depth

#define WARPS_M (BM/WM) // = 2. Rows per block devided by rows per warp is the number of warps in a block
#define WARPS_N (BN/WN) // = 2. same thing but for cols
#define FRAG_M (WM/16)  // Number of fragments per warp-row (2)
#define FRAG_N (WN/16)  // Number of fragments per warp-col (2)

#define T 128

// FRAG_M * FRAG_N = 4 --> there are 4 accumulator fragments per warp

using namespace nvcuda;

__global__ void fp16_gemm_kernel_warp_tiled_smem(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // Things this kernel needs to do
    // 1. Work out which output warp-block this warp is responsible for
    // 2. 
    int warpRow = threadIdx.y / WARPS_N;
    int warpCol = threadIdx.y % WARPS_N;

    int t = threadIdx.y * blockDim.x + threadIdx.x;       // flattened thread index 0...127

    // This blocks origin in C, element-level
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // We will cooperatively load sA and sB in the K-loop and cooperatively write results to sC
    __shared__ half sA[BM][BK];
    __shared__ half sB[BK][BN];
    __shared__ float sC[BM][BN];

    // Fragment arrays
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FRAG_M][FRAG_N];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[FRAG_M];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[FRAG_N];

    // We need to initialize each acc fragment with zeros
    // Note that we need to unroll this loop or else the compiler will
    // place the fragments in locally memory to make them indexable
    // If we unroll, each index is converted to an actual register ID by the compiler
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const half zero = __float2half(0.0f);
    // Main k-loop
    for (int k = 0; k < K; k += BK) {
        // Stage A: BM x BK = 1024 elements
        // With 128 threads, this loops 16 times

        // So warp (0, 0) will want to read sA[0...32][:] and sB[:][0...32], etc
        #pragma unroll
        for (int e = t; e < BM*BK; e += T) {
            // compose e into coordinates within a block
            // Note that sA will be 2D matrix of BM*BK
            int r = e / BK;
            int c = e % BK;
            int globR = blockRow + r;
            int globC = k + c;
            sA[r][c] = (globR < M && globC < K) ? A[globR * K + globC] : zero;
        }

        // Load sB BK x BN = 16 * 64 
        #pragma unroll
        for (int e = t; e < BK*BN; e += T) {
            int r = e / BN;
            int c = e % BN;
            int globR = k + r;
            int globC = blockCol + c;
            sB[r][c] = (globR < K && globC < N) ? B[globR * N + globC] : zero;
        }

        __syncthreads();

        // Each warp loads the relevant frags from smem
        #pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
            wmma::load_matrix_sync(a_frag[i], &sA[warpRow * WM + i*16][0], BK);
        }

        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::load_matrix_sync(b_frag[j], &sB[0][warpCol * WN + j*16], BN);
        }

        #pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAG_N; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }

        __syncthreads();
    }

    // Epilogue
    // Recall each warp has a (2,2) fragment-level block that it needs to write
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::store_matrix_sync(&sC[warpRow * WM + i * 16][warpCol * WN + j * 16], acc[i][j], BN, wmma::mem_row_major);
        }
    }

    __syncthreads();
    
    // Now we want to load from C, and to the scaling part of the GEMM
    #pragma unroll
    for (int e = t; e < BM*BN; e += T) {
        // Convert e to coords within block
        int r = e / BN;
        int c = e % BN;
        int globR = blockRow + r;
        int globC = blockCol + c;
        if (globR < M && globC < N) {
            float c_value = __half2float(C[globR * N + globC]);
            C[globR * N + globC] = __float2half(alpha * sC[r][c] + beta * c_value);
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {

    dim3 threadsPerBlock(32, 4);    // Still (32, 4), threadIdx.y represent the 2x2 warp grid

    // It is no longer one thread per output row
    dim3 blocksPerGrid((N + BN - 1)/ BN, (M + BM - 1)/ BM);

    // Launch on default stream
    fp16_gemm_kernel_warp_tiled_smem<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);

    cudaDeviceSynchronize();
}

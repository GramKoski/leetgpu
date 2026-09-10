#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#define WARPS_M 1
#define WARPS_N 4

#define BM (WARPS_M * 16)
#define BN (WARPS_N * 16)
#define BK 16

using namespace nvcuda;

__global__ void fp16_gemm_kernel_warp_tiled_smem(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    
}

__global__ void fp16_gemm_kernel_smem(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // wmma gemm kernel with cooperative staging in smem
    // Unlike the kernel below, we guard against M, N, K that aren't multiples of 16 via padding
    __shared__ half sA[BM][BK];

    __shared__ half sB[BK][BN];
    __shared__ float sC[BM][BN];

    int t = threadIdx.y * blockDim.x + threadIdx.x; // flattened thread index
    int T = blockDim.x * blockDim.y;     //threadsPerBlock = 128

    int blockRow = blockIdx.y * BM;     // block-level row of A that this block works on
    int blockCol = blockIdx.x * BN;     // block-level column of B that this block works 
    // remember that each block-level column corresponds to 4 tile-level columns (1, 4) tiles per block

    int warpCol = threadIdx.y % WARPS_N; // Which tile-column within block for this warp

    // Declare warp fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

    // Initialize accumulator with 0
    wmma::fill_fragment(acc, 0.0f);
    
    // K loop
    // This is the load and multiply loop
    // Similar idea as the kernel below but instead of simply loading
    // relevant tile directly into warps registers we cooperatively load it into smem

    for (int k = 0; k < K; k += BK) {
        // This is the core of the cooperative staging
        // We transform the flattened tid into tile local coordinates
        // and do a T-stride loop to fill the smem buffer

        // 1. Load a block of A (BM*BK)
        // T-stride loop. Each iteration we jump the number of threads in a block
        // We do this until we fill up the smem
        // Notice that this make the gmem access coallesced because as supposed
        // to having one thread load a contiguous segment in serial
        // BM * BK = 256, 256/T = 2. So two loop iterations
        for (int e = t; e < BM*BK; e += T) {
            // Compose e into coordinates within block
            int r = e / BK;
            int c = e % BK;

            // Get the global row and column of A
            int globR = blockRow + r;
            int globC = k + c;

            // Load element from gmem, fill with zero if out-of-bounds
            sA[r][c] = (globR < M && globC < K) ? A[globR * K + globC] : __float2half(0.0f);
        }

        // We do the same thing to load B
        // BK*BN = 16 * 64 = 1024, 1024 / T = 1024 / 128 = 8
        // So we have 8 iterations of this loop
        for (int e = t; e < BK * BN; e += T) {
            int r = e / BN;
            int c = e % BN;
            int globR = k + r;
            int globC = blockCol + c;
            sB[r][c] = (globR < K && globC < N) ? B[globR * N + globC] : __float2half(0.0f);
        }

        // Make sure sA and sB are fully loaded
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &sA[0][0], BK);  // Same A-tile for every warp
        wmma::load_matrix_sync(b_frag, &sB[0][warpCol * 16], BN);    // Depends on warpCol (threadIdx.y)
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();    // barrier before next iteration overwrites sA and sB;
    }


    // Now we cooperatively write to sC and then store in gmem in order to avoid out-of-bounds writes
    // Note it is a bit different this time because we are writing to 
    wmma::store_matrix_sync(&sC[0][warpCol * 16], acc, BN, wmma::mem_row_major);     // Store c_frag to this warp's cTIle
    __syncthreads();    // Make sure that all warps have written their tile to sC

    for (int e = t; e < BM*BN; e += T) {
        int r = e / BN;
        int c = e % BN;
        int globR = blockRow + r;
        int globC = blockCol + c;
        if (globR < M && globC < N) {
            float c_value = __half2float(C[globR * N + globC]);
            C[(globR * N) + globC] = __float2half(alpha * sC[r][c] + beta * c_value);
        }
    }
}

__global__ void fp16_gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // Note that in this kernel, M, N, and K have to be a multiple of 16
    // This won't work for leetgpu test cases.
    int tid = threadIdx.x;
    
    int tileM = blockIdx.y * WARPS_M;   // Tile-wise row of A that this warp is responsible for
    int tileN = blockIdx.x * WARPS_N + threadIdx.y;    // Tile-wise column of B that this warp is responsible for


    if (tileM * 16 >= M || tileN * 16 >= N) return;

    // Declare warp fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

    // Initialize accumulator with 0
    wmma::fill_fragment(acc, 0.0f);
    
    // K-loop: We want to do two things
    // 1. Load the tile from hbm (addressed by top-left "origin") into a_frag and b_frag
    // 2. Execute wmma instruction
    for (int k = 0; k < K; k += 16) {
        // Pointers to the origin of the A and B tile
        const half* aTile = A + (tileM * 16) * K + k;   // row = tileM*16, col = k
        const half* bTile = B + (tileN * 16) + (k * N); // row = k*N, col = tileN * 16

        wmma::load_matrix_sync(a_frag, aTile, K);    // ldm = K (strid K to get next tile-row)
        wmma::load_matrix_sync(b_frag, bTile, N);    // ldm = N (stride N to get next tile-row)
        wmma::mma_sync(acc, a_frag, b_frag, acc);    // acc = acc + a_frag*b_frag
    }
    // Now we need to do some things:
    // Get original tile of C (to do beta * C_initial)
    // Store it in c_frag
    half* cTile = C + (tileM * 16) * N + tileN * 16; // row = tileM*16, col = tileN*16
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // Load the C tile into c_frag
    // Note that we have to give an argument specifying the layout of the incoming tensor
    // We have to do the same when we store the accumulator in HBM
    // This is because wmma::accumulator types have no inherent layout
    // They can be loaded in different layout than they are stored
    wmma::load_matrix_sync(c_frag, cTile, N, wmma::mem_row_major);

    // Now note that acc.num_elements is 8 (8 elements in each thread-local register lane)
    for (int i = 0; i < acc.num_elements; ++i) {
        c_frag.x[i] = __float2half(alpha * acc.x[i] + beta * __half2float(c_frag.x[i]));   // Pay close attention dtypes throughout this equation
    }

    wmma::store_matrix_sync(cTile, c_frag, N, wmma::mem_row_major);     // Store c_frag to this warp's cTIle
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // Block shape
    // Note that blockDim.x is chosen because we have 32 threads in a warp
    // and blockDim.y = 4 is chosen to saturate SM occcupancy

    // Each warp is doing a 16x16x16 matmul, which corresponds to one 16x16 tile of the output
    // Therefore, in this case each block corresponds to 16x64 tile of output C
    dim3 threadsPerBlock(32, 4);
    dim3 blocksPerGrid((N + WARPS_N*16 -1)/ (WARPS_N*16), (M + WARPS_M * 16 - 1)/ (WARPS_M * 16));

    // Launch on default stream
    fp16_gemm_kernel_smem<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);

    cudaDeviceSynchronize();
}

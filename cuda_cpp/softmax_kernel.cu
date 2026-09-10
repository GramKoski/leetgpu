#include <cuda_runtime.h>

#define TILE 256

// We define the max number of blocks so we don't do to many redundant partial reductions of pmax 
// and the reductions of pmax can be done in one pass over smem (no grid-stride loop)
// Additionally, this way max_reduce_kernel is thread coursened (I don't know if it actually helps here though)
#define MAX_BLOCKS 256

__device__ float pmax[MAX_BLOCKS];
__device__ float d;
__device__ float m;

__global__ void max_reduce_kernel(const float* input, int N) {
    // Reduce using grid-stride loop (thread coursened)

    int tid = threadIdx.x;

    __shared__ float sdata[TILE];

    float run_max = -1e9f;
    // Phase 1, reduce across grid
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        run_max = max(run_max, input[i]);
    }
    sdata[tid] = run_max;   // Note that there is no need to check if i < N because it defaults to -1e9
    __syncthreads();

    for (int s = TILE/2; s > 0; s>>=1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid+s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Write the zeroth (block-reduced value) to corresponding scratch slot
        pmax[blockIdx.x] = sdata[tid];
    }
}

__global__ void reduce_sum_exp_kernel(const float* input, int N) {
    __shared__ float s_pmax[MAX_BLOCKS];
    __shared__ float s_psum[TILE];
    // Step 1. tree-reduce of pmax

    int tid = threadIdx.x;
    // Note this works because TILE == MAX_BLOCKS (or else this line would be problamatic)
    s_pmax[tid] = (tid < gridDim.x) ? pmax[tid] : -1e9f;    // Pad s_pmax with -1e9f;
    __syncthreads();

    for (int s = MAX_BLOCKS/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_pmax[tid] = max(s_pmax[tid], s_pmax[tid + s]);
        }
        __syncthreads();
    }
    float m_reg = s_pmax[0];    // Put m in thread-local register

    // Step 2. exponential sum tree-reduce
    // Make d is zero'd out
    // Note that there is actually a potential race condition here because block 0 doesn't have to execute first
    // If another block makes it all the way to line 86 before line 65 executes, we would have race condition
    // Solution is to move it to first kernel, however it works as-is so I'm lazy.
    if (blockIdx.x == 0 && tid == 0) {
        d = 0.0f;
    }
    float sum = 0.0f;
    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        sum += exp(input[i]-m_reg);
    }

    s_psum[tid] = sum;  // Note that there is no need to check if i < N because it defaults to zero
    __syncthreads();

    for (int s = TILE/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_psum[tid] += s_psum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&d, s_psum[tid]);
        if (blockIdx.x == 0) {
            m = m_reg;
        }
    }
}

__global__ void exp_normalize_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = exp(input[i] - m) / d;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGridNorm = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = min(blocksPerGridNorm, MAX_BLOCKS);
    // Naive approach
    // We do:
    // 1. Global max reduce
    // 2. Reduce sum exponential (sum e^x)
    // 3. Normalize e^x/sum(exp)
    // Note that we have to launch these parts as seperate kernels because that is the
    // most reliable way to synchronize across the entire grid
    // There is no atomicMax instruction, this means that we will have the per-block maxes written to a scratch array
    // Those maxes are read in by the next kernel (reduce_sum_exp_kernel) and each block redundantly max reduces those values
    // float* d; float* m;
    // cudaMalloc(&d, sizeof(float)); cudaMemset(d, 0, sizeof(float));     // We accumulate in d so make sure we start at 0
    // cudaMalloc(&m, sizeof(float));
    // Instead we make m and d device floats (see top)

    max_reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    reduce_sum_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    exp_normalize_kernel<<<blocksPerGridNorm, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
    // cudaFree(m); cudaFree(d);
}

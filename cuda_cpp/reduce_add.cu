#include <cuda_runtime.h>

#define TILE 256

__global__ void reduce_add_kernel(float* input, float* output, int N) {
    // Naive kernel
    // TODO: We could instead use thead coursening (less blocks) and accumulate multiple input elements in a thread in serial. 
    // This would allow for less tree adds (pure overhead adds)
    // It would also allow for less atomicAdds to the output
    __shared__ float sdata[TILE];

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + tid;

    // Load a tile of input into smem, pad out to end of tile
    sdata[tid] = (i < N) ? input[i] : 0.0f;

    __syncthreads();

    // In-kernel tree reduction 
    for (int s = TILE/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // Must make sure this level is fully reduced on all threads
        __syncthreads();
    }
    
    // sdata[0] has the block-level reduced sum
    // Different blocks could cause race condition--therefore we must do atomicAdd(float)
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // Zero the output pointer
    cudaMemset(output, 0, sizeof(float));
    int threadsPerBlock = TILE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch on default stream
    reduce_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();

}

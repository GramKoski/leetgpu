#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    // I am not sure if there is any advantage to using smem here
    // In the transpose kernel, we used smem for memory coalescing,
    // but the reads and writes here are already coalesced
    // Let's just do naive kernel
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < N/2) {
        float element = input[x];
        input[x] = input[N-1-x];
        input[N-1-x] = element;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}

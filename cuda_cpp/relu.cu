#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    // Very simple kernel
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < N) {
        float a = input[x];
        output[x] = (a > 0.0) ? a : 0.0f;   // Relu function
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
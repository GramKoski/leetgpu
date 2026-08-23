#include <cuda_runtime.h>

#define TILE 256

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    // Each thread is responsible for one output
    // We iterate 0->kernel_size -1
    // We want kernel is smem to increase bandwidth
    extern __shared__ float kernel_shared[];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int ld_tiles = (kernel_size + TILE - 1) / TILE;
    for (int tileIdx = 0; tileIdx < ld_tiles; ++tileIdx) {
        if (tileIdx * TILE + threadIdx.x < kernel_size) {
            kernel_shared[tileIdx * TILE + threadIdx.x] = kernel[tileIdx * TILE + threadIdx.x];
        }
    }
    __syncthreads();    // Make sure the kernel is loaded in smem

    int output_size = input_size - kernel_size + 1;
    float sum = 0.0;

    // Each thread is responsible for one output element
    for (int kernelIdx = 0; kernelIdx < kernel_size; ++kernelIdx) {
        if (x < output_size) {
            sum += input[x + kernelIdx] * kernel_shared[kernelIdx];
        }
    }
    if (x < output_size) { 
        output[x] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;



    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, kernel_size * sizeof(float)>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
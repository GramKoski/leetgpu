#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    // Naive approach
    // Possible inneficiencies: 
    // Memory isn't perfectly coalesced (each thread in a warp skips 3 bytes)
    // We only treat x as the thread-verying dimension, so we suffer in cases
    // where width is small and height is large
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (xIdx < width) {
        for (int yIdx = 0; yIdx < height; ++yIdx) {
            for (int colorIdx = 0; colorIdx < 3; ++ colorIdx) {
                image[(yIdx*width + xIdx)*4 + colorIdx] = 255-image[(yIdx*width + xIdx)*4 + colorIdx];
            }
        }
    }


}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    // We are working with a one dimensional thread block
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}


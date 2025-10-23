#include <iostream>
#include <math.h>
#include <chrono>
#include <iomanip>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
 int N = 1<<29;
 float *x, *y;

 // Allocate Unified Memory – accessible from CPU or GPU
 cudaMallocManaged(&x, N*sizeof(float));
 cudaMallocManaged(&y, N*sizeof(float));

 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }

 int deviceID=0;
 cudaMemPrefetchAsync((const void *)x, N*sizeof(float), deviceID) ;
 cudaMemPrefetchAsync((const void *)y, N*sizeof(float), deviceID) ;

 std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

 int blockSize = 256;
 int numBlocks = (N + blockSize - 1) / blockSize;
 // Run kernel on 1M elements on the GPU
 add<<<numBlocks, blockSize>>>(N, x, y);

 // Wait for GPU to finish before accessing on host
 cudaDeviceSynchronize();

 std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
 std::chrono::duration<double> elapsed = end_time - start_time;
 std::cout << " Elapsed time is : " << std::setprecision(5) << elapsed.count() << " (sec) " << std::endl;

 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++) {
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 }
 std::cout << "Max error: " << maxError << std::endl;

 // Free memory
 cudaFree(x);
 cudaFree(y);
  return 0;
}

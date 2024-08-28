// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include "../wb.h"
#include "solution.h"
#include <iostream>
using namespace std;

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add_elementwise(float *array, float *add, int len) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<len) array[i] += add[blockIdx.x];
}

__global__ void scan_Kogge_Stones(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  __shared__ float B1[BLOCK_SIZE];
  __shared__ float B2[BLOCK_SIZE];
  float* source = &B1[0];
  float* dest = &B2[0];
  float* temp;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int index;
  unsigned int t = threadIdx.x;
  B1[threadIdx.x] = input[i];
  
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (t < stride) {
      dest[t] = source[t];
    }
    index = t + stride;
    if (index < blockDim.x) {
      dest[index] = source[index] + source[t];
    }
    temp = dest; dest = source; source = temp;
  }

  __syncthreads();
  output[i] = source[t];
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numElements - 1) / BLOCK_SIZE + 1, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan_Kogge_Stones<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);
  
  float temp[(numElements - 1) / BLOCK_SIZE + 1];
  temp[0]=0;
  for (int i=1; i<(numElements - 1) / BLOCK_SIZE + 1; i++) {
    temp[i] = hostOutput[i * BLOCK_SIZE - 1] + temp[i - 1];
  }

  float* temp_device;
  cudaMalloc((void **)&temp_device, ((numElements - 1) / BLOCK_SIZE + 1) * sizeof(float));
  cudaMemcpy(temp_device, temp, ((numElements - 1) / BLOCK_SIZE + 1) * sizeof(float), cudaMemcpyHostToDevice);

  add_elementwise<<<DimGrid, DimBlock>>>(deviceOutput, temp_device, numElements);
  cudaDeviceSynchronize();
  cudaFree(temp_device);
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

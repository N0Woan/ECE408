#include "../wb.h"
#include "solution.h"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace std;

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 10

//@@ Define constant memory for device kernel here
__constant__ float kernel[27];

__global__ void conv3d(float *input, float *output, const int z_size, const int y_size, const int x_size, int mask_width) {
  //@@ Insert kernel code here
  __shared__ float input_ds[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int radius = mask_width / 2;
  
  int row_o = blockIdx.y * (TILE_WIDTH - (mask_width - 1)) + ty;
  int col_o = blockIdx.x * (TILE_WIDTH - (mask_width - 1)) + tx;
  int dep_o = blockIdx.z * (TILE_WIDTH - (mask_width - 1)) + tz;

  int row_i = row_o - radius;
  int col_i = col_o - radius;
  int dep_i = dep_o - radius;

  // Load input data to shared memory
  if (row_i >= 0 && row_i < y_size && col_i >= 0 && col_i < x_size && dep_i >= 0 && dep_i < z_size) {
    input_ds[tz][ty][tx] = input[dep_i * y_size * x_size + row_i * x_size + col_i];
  } else {
    input_ds[tz][ty][tx] = 0.0f;
  }

  __syncthreads();

  // Compute output
  if (row_o < y_size && col_o < x_size && dep_o < z_size && tx < (TILE_WIDTH - (mask_width - 1)) && ty < (TILE_WIDTH - (mask_width - 1)) && tz < (TILE_WIDTH - (mask_width - 1))) {
    float value = 0.0f;
    for (int i = 0; i < mask_width; i++) {
      for (int j = 0; j < mask_width; j++) {
        for (int k = 0; k < mask_width; k++) {
          value += kernel[i * mask_width * mask_width + j * mask_width + k] * input_ds[i + tz][j + ty][k + tx];
        }
      }
    }
    output[dep_o * y_size * x_size + row_o * x_size + col_o] = value;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  int mask_width;

  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
  wbTime_start(GPU, "Doing GPU memory allocation");

  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");
  wbTime_start(Copy, "Copying data to the GPU");

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kernel, hostKernel, kernelLength * sizeof(float));

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  mask_width = int(pow(kernelLength, 1.0/3.0));
  dim3 DimGrid((x_size - 1) / (TILE_WIDTH-(mask_width-1)) + 1, (y_size - 1) / (TILE_WIDTH-(mask_width-1)) + 1, (z_size - 1) / (TILE_WIDTH-(mask_width-1)) + 1);
  cout << (x_size - 1) / (TILE_WIDTH-(mask_width-1)) + 1 << "bruhbruh\n";
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size, mask_width);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float),
             cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void reductionWithCudaImproved(float* result, const float* input);
__global__ void reductionKernelImproved(float* result, const float* input);
void reductionCPU(float* result, const float* input);

#define SIZE 1000000
#define TILE 32
#define ILP 8
#define BLOCK_X_IMPR (TILE / ILP)
#define BLOCK_Y_IMPR 32
#define BLOCK_COUNT_X_IMPR 100


void reductionCPU(float* result, const float* input)
{
    for (int i = 0; i < SIZE; i++)
        *result += input[i];
}

__global__ void reductionKernelImproved(float* result, const float* input)
{
    int i;
    int col = (blockDim.x * blockIdx.x + threadIdx.x) * ILP;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int index = row * blockDim.x * gridDim.x * ILP + col;
    __shared__ float interResult;

    if (threadIdx.x == 0 && threadIdx.y == 0)
        interResult = 0.0;

    __syncthreads();

#pragma unroll ILP
    for (i = 0; i < ILP; i++)
    {
        if (index < SIZE)
        {
            atomicAdd(&interResult, input[index]);
            index++;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(result, interResult);
}

void reductionWithCudaImproved(float* result, const float* input)
{
    dim3 dim_grid, dim_block;

    float* dev_input = 0;
    float* dev_result = 0;
    cudaEvent_t start, stop;
    float elapsed = 0;
    double gpuBandwidth;

    dim_block.x = BLOCK_X_IMPR;
    dim_block.y = BLOCK_Y_IMPR;
    dim_block.z = 1;

    dim_grid.x = BLOCK_COUNT_X_IMPR;
    dim_grid.y = (int)ceil((float)SIZE / (float)(TILE * dim_block.y * BLOCK_COUNT_X_IMPR));
    dim_grid.z = 1;

    cudaSetDevice(0);

    cudaMalloc((void**)&dev_input, SIZE * sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float));
    cudaMemcpy(dev_input, input, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_result, result, sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reductionKernelImproved << <dim_grid, dim_block >> > (dev_result, dev_input);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    printf("GPU Time (improved): %f ms\n", elapsed);

    cudaDeviceSynchronize();

    cudaMemcpy(result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_result);

    return;
}


int main()
{
    int i;
    float* input;
    float resultCPU, resultGPU;
    double cpuTime, cpuBandwidth;
    printf("Size : %d \n", SIZE);
    input = (float*)malloc(SIZE * sizeof(float));
    resultCPU = 0.0;
    resultGPU = 0.0;

    srand((int)time(NULL));

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    for (i = 0; i < SIZE; i++)
        input[i] = rand() % 10 - 5;

    start = std::chrono::high_resolution_clock::now();
    reductionCPU(&resultCPU, input);
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    cpuTime = (diff.count() * 1000);
    printf("CPU Time: %f ms\n", cpuTime);

    reductionWithCudaImproved(&resultGPU, input);

    if (resultCPU != resultGPU)
        printf("CPU result does not match GPU result \n\n");
    else
        printf("CPU result matches GPU result\n\n");

    return 0;
}
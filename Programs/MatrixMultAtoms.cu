#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 16

__global__ void matrixMult(float *A, float *B, float *C, int size);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s size\n", argv[0]);
        exit(1);
    }

    int size = atoi(argv[1]);
    float input_size = size * size * sizeof(float);
    if (size <= 0)
    {
        printf("Invalid matrix size: %d\n", size);
        exit(1);
    }

    // Allocate memory for matrices A, B, and C on the host
    float *A = (float *)malloc(input_size);
    float *B = (float *)malloc(input_size);
    float *C = (float *)malloc(input_size);

    // Allocate memory for matrices A, B, and C on the device
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, input_size);
    cudaMalloc(&dev_b, input_size);
    cudaMalloc(&dev_c, input_size);

    // Load matrices A and B with random numbers
    srand(42);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = (float)rand() / (float)RAND_MAX;
            B[i * size + j] = (float)rand() / (float)RAND_MAX;
        }
    }

    // Copy matrices A and B from host to device
    cudaMemcpy(dev_a, A, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, B, input_size, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions for the MatrixMultKernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Create CUDA events to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call the MatrixMultKernel on the device
    cudaEventRecord(start);
    matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, size);
    cudaEventRecord(stop);

    // Copy matrix C from device to host
    cudaMemcpy(C, dev_c, input_size, cudaMemcpyDeviceToHost);

    // Calculate the elapsed time in seconds
    float elapsedTime;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the execution time
    printf("Execution time: %f ms\n", elapsedTime);
     FILE *csv_file;
    char csv_filename[100];
    sprintf(csv_filename, "CSV/Atomics.csv");
    csv_file = fopen(csv_filename, "a");
    fprintf(csv_file, "%d,%f\n", size, elapsedTime);
    fclose(csv_file);

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

__global__ void matrixMult(float *A, float *B, float *C, int size)
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (size + BLOCK_SIZE - 1) / BLOCK_SIZE; t++)
    {
        int tiledRow = blockIdx.y * blockDim.y + threadIdx.y;
        int tiledCol = t * blockDim.x + threadIdx.x;

        if (tiledRow < size && tiledCol < size)
            tileA[threadIdx.y][threadIdx.x] = A[tiledRow * size + tiledCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;

        tiledRow = t * blockDim.y + threadIdx.y;
        tiledCol = blockIdx.x * blockDim.x + threadIdx.x;

        if (tiledRow < size && tiledCol < size)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * size + tiledCol];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < size && col < size)
        atomicAdd(&C[row * size + col], sum);
}

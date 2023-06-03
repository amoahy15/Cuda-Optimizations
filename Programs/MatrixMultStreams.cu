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

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Copy matrices A and B from host to device asynchronously
    cudaMemcpyAsync(dev_a, A, input_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_b, B, input_size, cudaMemcpyHostToDevice, stream2);

    // Define the grid and block dimensions for the MatrixMultKernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Create CUDA events to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call the MatrixMultKernel on the device asynchronously
    cudaEventRecord(start);
    matrixMult<<<dimGrid, dimBlock, 0, stream1>>>(dev_a, dev_b, dev_c, size);
    cudaEventRecord(stop, stream1);

    // Copy matrix C from device to host asynchronously
    cudaMemcpyAsync(C, dev_c, input_size, cudaMemcpyDeviceToHost, stream2);

    // Calculate the elapsed time in milliseconds
    float elapsedTime;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the execution time
    printf("Execution time: %f ms\n", elapsedTime);
     FILE *csv_file;
    char csv_filename[100];
    sprintf(csv_filename, "CSV/Streams.csv");
    csv_file = fopen(csv_filename, "a");
    fprintf(csv_file, "%d,%f\n", size, elapsedTime);
    fclose(csv_file);

    // Free memory and destroy streams
    free(A);
    free(B);
    free(C);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}

__global__ void matrixMult(float *A, float *B, float *C, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k;
    if (row < size && col < size)
    {
        float sum = 0.0f;
        for (k = 0; k < size; k++)
        {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}
%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define SCALAR 2.0f

/* =====================================================
   CUDA ядра
   ===================================================== */

// a) Только global memory
__global__ void kernel_global(float* d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_array[idx] *= SCALAR;
}

// b) Global + shared memory
__global__ void kernel_shared(float* d_array, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n)
        sdata[tid] = d_array[idx];
    __syncthreads();

    if (idx < n)
        sdata[tid] *= SCALAR;
    __syncthreads();

    if (idx < n)
        d_array[idx] = sdata[tid];
}

// c) Local memory (регистры)
__global__ void kernel_local(float* d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = d_array[idx]; // локальная переменная
        temp *= SCALAR;
        d_array[idx] = temp;
    }
}

/* =====================================================
   Функция измерения времени
   ===================================================== */
float measure_time(void (*kernel)(float*, int), float* d_array, int n, dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_array, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

/* =====================================================
   MAIN
   ===================================================== */
int main() {
    int sizes[] = {10000, 100000, 1000000};
    int num_sizes = 3;

    for (int i = 0; i < num_sizes; i++) {
        int N = sizes[i];
        float* h_array = (float*)malloc(N * sizeof(float));
        float* d_array;
        cudaMalloc(&d_array, N * sizeof(float));

        // Генерация случайных данных
        for (int j = 0; j < N; j++) h_array[j] = (float)(rand() % 1000);
        cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(BLOCK_SIZE);
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // a) Global
        float t_global = measure_time(kernel_global, d_array, N, grid, block);
        cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

        // b) Shared
        float t_shared = measure_time(kernel_shared, d_array, N, grid, block);
        cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

        // c) Local
        float t_local = measure_time(kernel_local, d_array, N, grid, block);
        cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

        printf("Array size: %d\n", N);
        printf("Global memory time: %.3f ms\n", t_global);
        printf("Shared memory time: %.3f ms\n", t_shared);
        printf("Local memory time:  %.3f ms\n\n", t_local);

        free(h_array);
        cudaFree(d_array);
    }
    return 0;
}

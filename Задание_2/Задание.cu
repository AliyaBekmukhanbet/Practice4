%%cuda
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define BLOCK_SIZE 256

/* ============================================================
   Генерация случайных чисел на GPU
   ============================================================ */
__global__ void generate_random(float* data, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_uniform(&state);
    }
}

/* ============================================================
   a) РЕДУКЦИЯ ТОЛЬКО С GLOBAL MEMORY (НЕОПТИМИЗИРОВАННАЯ)
   ============================================================ */
__global__ void reduce_global(float* data, int stride, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + stride < n && idx % (2 * stride) == 0) {
        data[idx] += data[idx + stride];
    }
}

/* ============================================================
   b) РЕДУКЦИЯ С SHARED MEMORY (ОПТИМИЗИРОВАННАЯ)
   ============================================================ */
__global__ void reduce_shared(float* d_in, float* d_out, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? d_in[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

/* ============================================================
   MAIN
   ============================================================ */
int main() {
    float *d_data, *d_temp;
    float h_result;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* ---------------- Генерация данных ---------------- */
    generate_random<<<blocks, BLOCK_SIZE>>>(d_data, time(NULL));
    cudaDeviceSynchronize();

    /* ===================================================
       a) GLOBAL MEMORY REDUCTION
       =================================================== */
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);

    for (int stride = 1; stride < N; stride *= 2) {
        reduce_global<<<blocks, BLOCK_SIZE>>>(d_data, stride, N);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    cudaMemcpy(&h_result, d_data, sizeof(float), cudaMemcpyDeviceToHost);

    float time_global;
    cudaEventElapsedTime(&time_global, start1, stop1);

    printf("GLOBAL MEMORY SUM = %f\n", h_result);
    printf("GLOBAL MEMORY TIME = %.3f ms\n\n", time_global);

    /* ===================================================
       b) SHARED MEMORY REDUCTION
       =================================================== */
    generate_random<<<blocks, BLOCK_SIZE>>>(d_data, time(NULL));
    cudaDeviceSynchronize();

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    int n = N;
    float *input = d_data;
    float *output = d_temp;

    cudaEventRecord(start2);

    while (n > 1) {
        int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce_shared<<<grid, BLOCK_SIZE>>>(input, output, n);
        cudaDeviceSynchronize();

        n = grid;
        float* tmp = input;
        input = output;
        output = tmp;
    }

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    cudaMemcpy(&h_result, input, sizeof(float), cudaMemcpyDeviceToHost);

    float time_shared;
    cudaEventElapsedTime(&time_shared, start2, stop2);

    printf("SHARED MEMORY SUM = %f\n", h_result);
    printf("SHARED MEMORY TIME = %.3f ms\n\n", time_shared);

    /* ===================================================
       Итог
       =================================================== */
    printf("ACCELERATION: %.2fx\n", time_global / time_shared);

    cudaFree(d_data);
    cudaFree(d_temp);

    return 0;
}

%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024        // Для примера (можно увеличить)
#define BLOCK_SIZE 256

/* ============================================================
   CUDA-ядро сортировки пузырьком внутри блока
   ============================================================ */
__global__ void bubble_sort_block(float* d_array, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int start = blockIdx.x * blockDim.x;

    // Загружаем подмассив в shared memory
    if (start + tid < n)
        sdata[tid] = d_array[start + tid];
    else
      //  sdata[tid] = FLT_MAX; // Для блоков, которые выходят за предел массива
    __syncthreads();

    // Пузырьковая сортировка в shared memory
    for (int i = 0; i < blockDim.x - 1; i++) {
        for (int j = 0; j < blockDim.x - i - 1; j++) {
            if (sdata[j] > sdata[j + 1]) {
                float tmp = sdata[j];
                sdata[j] = sdata[j + 1];
                sdata[j + 1] = tmp;
            }
        }
    }
    __syncthreads();

    // Копируем отсортированный блок обратно в global memory
    if (start + tid < n)
        d_array[start + tid] = sdata[tid];
}

/* ============================================================
   CUDA-ядро для слияния двух подмассивов с shared memory
   ============================================================ */
__global__ void merge_blocks(float* d_array, float* d_temp, int size, int n) {
    int start = 2 * blockIdx.x * size; // начало первого подмассива
    int mid = start + size;
    int end = start + 2 * size;

    if (mid >= n) mid = n;
    if (end >= n) end = n;

    extern __shared__ float sdata[]; // shared memory для двух подмассивов
    int tid = threadIdx.x;

    // Загружаем два подмассива в shared memory
    int total = end - start;
    if (tid < total) {
        if (tid + start < mid)
            sdata[tid] = d_array[start + tid];       // первый подмассив
        else
            sdata[tid] = d_array[mid + tid - (mid - start)]; // второй подмассив
    }
    __syncthreads();

    // Простое последовательное слияние в shared memory
    if (tid == 0) {
        int i = 0, j = mid - start, k = 0;
        while (i < mid - start && j < total) {
            d_temp[start + k++] = (sdata[i] < sdata[j]) ? sdata[i++] : sdata[j++];
        }
        while (i < mid - start) d_temp[start + k++] = sdata[i++];
        while (j < total) d_temp[start + k++] = sdata[j++];
    }
    __syncthreads();

    // Копируем результат обратно в global memory
    if (tid < total)
        d_array[start + tid] = d_temp[start + tid];
}

/* ============================================================
   MAIN
   ============================================================ */
int main() {
    float* h_array = (float*)malloc(N * sizeof(float));
    float* d_array;
    float* d_temp;
    srand(time(NULL));

    // Генерация случайного массива
    for (int i = 0; i < N; i++) {
        h_array[i] = (float)(rand() % 1000);
    }

    cudaMalloc(&d_array, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    // ---------------- Сортировка подмассивов пузырьком ----------------
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bubble_sort_block<<<blocks, BLOCK_SIZE>>>(d_array, N);
    cudaDeviceSynchronize();

    // ---------------- Слияние блоков ----------------
    int merge_size = BLOCK_SIZE;
    while (merge_size < N) {
        int merge_blocks_count = (N + 2 * merge_size - 1) / (2 * merge_size);
        merge_blocks<<<merge_blocks_count, 2 * merge_size, 2 * merge_size * sizeof(float)>>>(d_array, d_temp, merge_size, N);
        cudaDeviceSynchronize();
        merge_size *= 2;
    }

    // ---------------- Копирование и вывод ----------------
    cudaMemcpy(h_array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Отсортированный массив (первые 20 элементов):\n");
    for (int i = 0; i < 20 && i < N; i++) {
        printf("%.2f ", h_array[i]);
    }
    printf("\n");

    cudaFree(d_array);
    cudaFree(d_temp);
    free(h_array);
    return 0;
}

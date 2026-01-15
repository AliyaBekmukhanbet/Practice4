%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define N 1000000
#define BLOCK_SIZE 256

// CUDA-ядро генерации случайных чисел
__global__ void generate_random(float* d_array, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Инициализация состояния генератора
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Генерация случайного числа в диапазоне (0, 1]
        d_array[idx] = curand_uniform(&state);
    }
}

int main() {
    float* d_array;
    float* h_array = (float*)malloc(N * sizeof(float));

    // Выделение памяти на GPU
    cudaMalloc((void**)&d_array, N * sizeof(float));

    // Настройка сетки CUDA
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Запуск CUDA-ядра
    generate_random<<<gridSize, BLOCK_SIZE>>>(d_array, time(NULL));

    // Копирование данных с GPU на CPU
    cudaMemcpy(h_array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Проверка первых элементов
    for (int i = 0; i < 5; i++) {
        printf("h_array[%d] = %f\n", i, h_array[i]);
    }

    // Освобождение памяти
    cudaFree(d_array);
    free(h_array);

    return 0;
}

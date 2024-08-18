#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // Include AVX header

#define SIZE 2048  // Size of the matrix and vector
#define ITERATIONS 1000  // Run the multiplication 100 times to increase runtime

void matrix_vector_mult_avx(float A[SIZE][SIZE], float x[SIZE], float y[SIZE]) {
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        for (int i = 0; i < SIZE; ++i) {
            __m256 sum = _mm256_setzero_ps();
            for (int j = 0; j < SIZE; j += 8) {  // Process 8 elements at a time
                __m256 a = _mm256_loadu_ps(&A[i][j]);
                __m256 b = _mm256_loadu_ps(&x[j]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            // Manually sum the elements in the AVX register
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            y[i] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        }
    }
}

int main() {
    float (*A)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));
    float *x = malloc(sizeof(float[SIZE]));
    float *y = malloc(sizeof(float[SIZE]));

    // Initialize matrix and vector with some values
    for (int i = 0; i < SIZE; ++i) {
        x[i] = 1.0f;  // Vector initialized to 1.0
        for (int j = 0; j < SIZE; ++j) {
            A[i][j] = 1.0f;  // Matrix initialized to 1.0
        }
    }

    matrix_vector_mult_avx(A, x, y);

    // Print the first element of the result to ensure computation
    printf("y[0] = %f\n", y[0]);

    free(A);
    free(x);
    free(y);

    return 0;
}
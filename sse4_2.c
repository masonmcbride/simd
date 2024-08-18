#include <stdio.h>
#include <stdlib.h>
#include <smmintrin.h>  // Include SSE4.1 and SSE4.2 header

#define SIZE 2048  // Increase the size of the matrix and vector to 2048
#define ITERATIONS 1000  // Run the multiplication 100 times to increase runtime

void matrix_vector_mult_sse4_2(float A[SIZE][SIZE], float x[SIZE], float y[SIZE]) {
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        for (int i = 0; i < SIZE; ++i) {
            __m128 sum = _mm_setzero_ps();
            for (int j = 0; j < SIZE; j += 4) {  // Process 4 elements at a time
                __m128 a = _mm_loadu_ps(&A[i][j]);
                __m128 b = _mm_loadu_ps(&x[j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }
            // Use _mm_hadd_ps to horizontally add pairs of elements in the vector
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            _mm_store_ss(&y[i], sum);
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

    matrix_vector_mult_sse4_2(A, x, y);

    // Print the first element of the result to ensure computation
    printf("y[0] = %f\n", y[0]);

    free(A);
    free(x);
    free(y);

    return 0;
}
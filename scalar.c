#include <stdio.h>
#include <stdlib.h>

#define SIZE 2048  // Size of the matrix and vector
#define ITERATIONS 1000  // Run the multiplication 100 times to increase runtime

void matrix_vector_mult_scalar(float A[SIZE][SIZE], float x[SIZE], float y[SIZE]) {
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        for (int i = 0; i < SIZE; ++i) {
            y[i] = 0;
            for (int j = 0; j < SIZE; ++j) {
                y[i] += A[i][j] * x[j];
            }
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

    matrix_vector_mult_scalar(A, x, y);

    // Print the first element of the result to ensure computation
    printf("y[0] = %f\n", y[0]);

    free(A);
    free(x);
    free(y);

    return 0;
}
#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK_LAST() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct Matrix
{
    int rows;
    int cols;
    float* data;
};

Matrix createRandMatrix(int row, int col);
Matrix createMatrix(int row, int col);
void freeMatrix(Matrix m);
void printMatrix(FILE* out, Matrix m);
void matmul(Matrix A, Matrix B, Matrix outMat);
void add(Matrix A, Matrix B, Matrix outMat);
Matrix loadMatrix(float* data, int row, int col);

// Activation Functions
void sigmoid(Matrix m);
void tanhInplace(Matrix m);
void relu(Matrix m);
void hadamard(Matrix A, Matrix B, Matrix outMat);
void copyMatrix(Matrix src, Matrix dest);
void subtract(Matrix A, Matrix B, Matrix outMat);
void scale(Matrix m, float scalar);
void fillMatrix(Matrix m, float value);
void initMatrixRandom(Matrix m, float min_val, float max_val);
void initMatrixXavier(Matrix m, int input_size);

// Backward pass operations
void sigmoidBackward(Matrix grad_out, Matrix sigmoid_output, Matrix grad_in);
void tanhBackward(Matrix grad_out, Matrix tanh_output, Matrix grad_in);
void reluBackward(Matrix grad_out, Matrix relu_input, Matrix grad_in);
void matmulBackwardA(Matrix grad_out, Matrix B, Matrix grad_A);  // dL/dA = dL/dOut * B^T
void matmulBackwardB(Matrix grad_out, Matrix A, Matrix grad_B);  // dL/dB = A^T * dL/dOut
void transposeMatmul(Matrix A, Matrix B, Matrix outMat);  // A^T * B
void matmulTranspose(Matrix A, Matrix B, Matrix outMat);  // A * B^T
void addInplace(Matrix A, Matrix B);  // A += B
void clipGradients(Matrix m, float max_norm);

#endif
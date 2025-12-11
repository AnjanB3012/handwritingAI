#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cmath>

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

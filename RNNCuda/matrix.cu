#include "matrix.h"

Matrix createRandMatrix(int row, int col)
{
    Matrix m;
    m.rows = row;
    m.cols = col;
    cudaMallocManaged(&m.data, row*col*sizeof(float));
    for(int i=0;i<row*col;i++)
    {
        m.data[i] = (float) (rand() % 2);
    }
    return m;
}

Matrix createMatrix(int row, int col)
{
    Matrix m;
    m.rows = row;
    m.cols = col;
    cudaMallocManaged(&m.data, row*col*sizeof(float));
    return m;
}

void freeMatrix(Matrix m)
{
    cudaFree(m.data);
}

void printMatrix(FILE* out, Matrix m)
{
    for(int i=0; i<m.rows;i++)
    {
        for(int j=0; j<m.cols;j++)
        {
            fprintf(out, "%f ", m.data[(i*m.cols)+j]);
        }
        fprintf(out, "\n");
    }
}

__global__ void matmulkernel(const float* A, const float* B, float* outMat, int M, int N, int K)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M && col<N)
    {
        float val = 0;
        for(int i=0;i<K;i++)
        {
            val += A[row*K+i]*B[i*N+col];
        }
        outMat[row*N+col] = val;
    }
}

void matmul(Matrix A, Matrix B, Matrix outMat)
{
    dim3 threads(16,16);
    dim3 blocks((B.cols+15)/16, (A.rows+15)/16);
    matmulkernel<<<blocks, threads>>>(A.data, B.data, outMat.data, A.rows, B.cols, A.cols);
    cudaDeviceSynchronize();
}

__global__ void addkernel(float* A, float* B, float* outMat, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        outMat[i] = A[i]+B[i];
    }
}

void add(Matrix A, Matrix B, Matrix outMat)
{
    int n = A.rows * A.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    addkernel<<<blocks, threads>>>(A.data, B.data, outMat.data, n);
    cudaDeviceSynchronize();
}

Matrix loadMatrix(float* data, int row, int col)
{
    Matrix m;
    m.rows = row;
    m.cols = col;
    cudaMallocManaged(&m.data, row*col*sizeof(float));
    for(int i=0;i<row*col;i++)
    {
        m.data[i] = data[i];
    }
    return m;
}

__global__ void sigmoidKernel(float* x, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        float v = x[i];
        x[i] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void tanhKernel(float* x, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        x[i] = tanhf(x[i]);
    }
}

__global__ void hadamardKernel(const float* A, const float* B, float* C, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        C[i] = A[i] * B[i];
    }
}

__global__ void copyKernel(const float* src, const float* dest, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        dest[i] = src[i];
    }
}

void sigmoid(Matrix m)
{
    int n = m.rows*m.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    sigmoidKernel<<<blocks, threads>>>(m.data, n);
    cudaDeviceSynchronize();
}

void tanhInplace(Matrix m)
{
    int n = m.rows*m.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    tanhKernel<<<blocks, threads>>>(m.data, n);
    cudaDeviceSynchronize();
}

void hadamard(Matrix A, Matrix B, Matrix outMat)
{
    int n = A.rows*A.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    hadamardKernel<<<blocks, threads>>>(A.data, B.data, outMat.data, n);
    cudaDeviceSynchronize();
}

void copyMatrix(Matrix src, Matrix dest)
{
    int n = src.rows*src.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    copyKernel<<<blocks, threads>>>(src.data, dest.data, n);
    cudaDeviceSynchronize();
}

__global__ void reluKernel(float* x, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

void relu(Matrix m)
{
    int n = m.rows*m.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    reluKernel<<<blocks, threads>>>(m.data, n);
    cudaDeviceSynchronize();
}

__global__ void subtractKernel(const float* A, const float* B, float* outMat, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        outMat[i] = A[i] - B[i];
    }
}

void subtract(Matrix A, Matrix B, Matrix outMat)
{
    int n = A.rows * A.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    subtractKernel<<<blocks, threads>>>(A.data, B.data, outMat.data, n);
    cudaDeviceSynchronize();
}

__global__ void scaleKernel(float* x, float scalar, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        x[i] *= scalar;
    }
}

void scale(Matrix m, float scalar)
{
    int n = m.rows*m.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    scaleKernel<<<blocks, threads>>>(m.data, scalar, n);
    cudaDeviceSynchronize();
}

__global__ void fillKernel(float* x, float value, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        x[i] = value;
    }
}

void fillMatrix(Matrix m, float value)
{
    int n = m.rows*m.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    fillKernel<<<blocks, threads>>>(m.data, value, n);
    cudaDeviceSynchronize();
}

__global__ void initRandomKernel(float* x, float min_val, float max_val, int n, unsigned int seed)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int local_seed = seed + tid;
        float r = (float)local_seed / 2147483647.0f; // Normalize to [0,1]
        x[i] = min_val + r * (max_val - min_val);
    }
}

void initMatrixRandom(Matrix m, float min_val, float max_val)
{
    int n = m.rows*m.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    unsigned int seed = (unsigned int)time(NULL);
    initRandomKernel<<<blocks, threads>>>(m.data, min_val, max_val, n, seed);
    cudaDeviceSynchronize();
}

__global__ void initXavierKernel(float* x, float scale, int n, unsigned int seed)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int local_seed = seed + tid;
        float u1 = (float)(local_seed % 1000) / 1000.0f;
        float u2 = (float)((local_seed * 1103515245 + 12345) % 1000) / 1000.0f;
        // Box-Muller transform approximation
        float z = sqrtf(-2.0f * logf(u1 + 1e-8f)) * cosf(2.0f * 3.14159f * u2);
        x[i] = z * scale;
    }
}

void initMatrixXavier(Matrix m, int input_size)
{
    int n = m.rows*m.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    float scale = sqrtf(2.0f / (float)input_size);
    unsigned int seed = (unsigned int)time(NULL);
    initXavierKernel<<<blocks, threads>>>(m.data, scale, n, seed);
    cudaDeviceSynchronize();
}

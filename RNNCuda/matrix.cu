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
    // Removed sync for performance - operations are serialized in default stream
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

__global__ void copyKernel(const float* src, float* dest, int n)
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
    // sync removed for perf
}

void tanhInplace(Matrix m)
{
    int n = m.rows*m.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    tanhKernel<<<blocks, threads>>>(m.data, n);
    // sync removed for perf
}

void hadamard(Matrix A, Matrix B, Matrix outMat)
{
    int n = A.rows*A.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    hadamardKernel<<<blocks, threads>>>(A.data, B.data, outMat.data, n);
    // sync removed for perf
}

void copyMatrix(Matrix src, Matrix dest)
{
    int n = src.rows*src.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    copyKernel<<<blocks, threads>>>(src.data, dest.data, n);
    // sync removed for perf
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
    // sync removed for perf
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
    // sync removed for perf
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
    // sync removed for perf
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
    // sync removed for perf
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
    // sync removed for perf
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
    // sync removed for perf
}

// ============ BACKWARD PASS OPERATIONS ============

__global__ void sigmoidBackwardKernel(const float* grad_out, const float* sig_out, float* grad_in, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        // d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
        float s = sig_out[i];
        grad_in[i] = grad_out[i] * s * (1.0f - s);
    }
}

void sigmoidBackward(Matrix grad_out, Matrix sigmoid_output, Matrix grad_in)
{
    int n = grad_out.rows * grad_out.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    sigmoidBackwardKernel<<<blocks, threads>>>(grad_out.data, sigmoid_output.data, grad_in.data, n);
    // sync removed for perf
}

__global__ void tanhBackwardKernel(const float* grad_out, const float* tanh_out, float* grad_in, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        // d(tanh)/dx = 1 - tanh(x)^2
        float t = tanh_out[i];
        grad_in[i] = grad_out[i] * (1.0f - t * t);
    }
}

void tanhBackward(Matrix grad_out, Matrix tanh_output, Matrix grad_in)
{
    int n = grad_out.rows * grad_out.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    tanhBackwardKernel<<<blocks, threads>>>(grad_out.data, tanh_output.data, grad_in.data, n);
    // sync removed for perf
}

__global__ void reluBackwardKernel(const float* grad_out, const float* relu_in, float* grad_in, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        grad_in[i] = (relu_in[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}

void reluBackward(Matrix grad_out, Matrix relu_input, Matrix grad_in)
{
    int n = grad_out.rows * grad_out.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    reluBackwardKernel<<<blocks, threads>>>(grad_out.data, relu_input.data, grad_in.data, n);
    // sync removed for perf
}

// A^T * B -> outMat  (A is MxK, B is MxN, outMat is KxN)
__global__ void transposeMatmulKernel(const float* A, const float* B, float* outMat, int M, int K, int N)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;  // row in output (0..K)
    int col = blockIdx.x*blockDim.x+threadIdx.x;  // col in output (0..N)
    if(row < K && col < N)
    {
        float val = 0.0f;
        for(int i = 0; i < M; i++)
        {
            val += A[i*K + row] * B[i*N + col];  // A^T[row,i] * B[i,col]
        }
        outMat[row*N + col] = val;
    }
}

void transposeMatmul(Matrix A, Matrix B, Matrix outMat)
{
    // A is MxK, B is MxN, output is KxN
    dim3 threads(16,16);
    dim3 blocks((B.cols+15)/16, (A.cols+15)/16);
    transposeMatmulKernel<<<blocks, threads>>>(A.data, B.data, outMat.data, A.rows, A.cols, B.cols);
    // sync removed for perf
}

// A * B^T -> outMat  (A is MxK, B is NxK, outMat is MxN)
__global__ void matmulTransposeKernel(const float* A, const float* B, float* outMat, int M, int K, int N)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;  // row in output (0..M)
    int col = blockIdx.x*blockDim.x+threadIdx.x;  // col in output (0..N)
    if(row < M && col < N)
    {
        float val = 0.0f;
        for(int i = 0; i < K; i++)
        {
            val += A[row*K + i] * B[col*K + i];  // A[row,i] * B^T[i,col] = A[row,i] * B[col,i]
        }
        outMat[row*N + col] = val;
    }
}

void matmulTranspose(Matrix A, Matrix B, Matrix outMat)
{
    // A is MxK, B is NxK, output is MxN
    dim3 threads(16,16);
    dim3 blocks((B.rows+15)/16, (A.rows+15)/16);
    matmulTransposeKernel<<<blocks, threads>>>(A.data, B.data, outMat.data, A.rows, A.cols, B.rows);
    // sync removed for perf
}

// dL/dA = dL/dOut * B^T  (grad_out is MxN, B is KxN, grad_A is MxK)
void matmulBackwardA(Matrix grad_out, Matrix B, Matrix grad_A)
{
    // grad_A = grad_out * B^T
    matmulTranspose(grad_out, B, grad_A);
}

// dL/dB = A^T * dL/dOut  (A is MxK, grad_out is MxN, grad_B is KxN)
void matmulBackwardB(Matrix grad_out, Matrix A, Matrix grad_B)
{
    // grad_B = A^T * grad_out
    transposeMatmul(A, grad_out, grad_B);
}

__global__ void addInplaceKernel(float* A, const float* B, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        A[i] += B[i];
    }
}

void addInplace(Matrix A, Matrix B)
{
    int n = A.rows * A.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    addInplaceKernel<<<blocks, threads>>>(A.data, B.data, n);
    // sync removed for perf
}

__global__ void clipGradientsKernel(float* x, float max_norm, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {
        if(x[i] > max_norm) x[i] = max_norm;
        else if(x[i] < -max_norm) x[i] = -max_norm;
    }
}

void clipGradients(Matrix m, float max_norm)
{
    int n = m.rows * m.cols;
    int threads = 128;
    int blocks = (n+threads-1)/threads;
    clipGradientsKernel<<<blocks, threads>>>(m.data, max_norm, n);
    // sync removed for perf
}

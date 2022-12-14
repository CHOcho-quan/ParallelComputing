// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

extern __shared__ _FTYPE_ sharmem[];

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
__global__ __inline__ void matMul(int N, _FTYPE_ *__restrict__ C, _FTYPE_ *__restrict__ A, _FTYPE_ *__restrict__ B) {
    /**
    * Tiled Matrix Multiplication in CUTLASS
    * BLOCK_N / BLOCK_M / BLOCK_K - block size n / m / k
    * THREAD_M / THREAD_N - thread block size M / N
    * GRID_SIZE - BLOCK_M * BLOCK_N / THREAD_M * THREAD_N
    * **/
    _FTYPE_ * __restrict__ sA = &sharmem[0];
    _FTYPE_ * __restrict__ sB = &sA[BLOCK_N * BLOCK_K];
    // __shared__ _FTYPE_ sA[BLOCK_N][BLOCK_K], sB[BLOCK_K][BLOCK_M];
    int x, y;
    register _FTYPE_ rA[THREAD_N], rB[THREAD_M]; // These should resides in registers
    register _FTYPE_ c[THREAD_N][THREAD_M] = { 0 };
    int tx = threadIdx.x / (BLOCK_M / THREAD_M), ty = threadIdx.x % (BLOCK_M / THREAD_M);
    int xA = blockIdx.x * BLOCK_N, yA = 0, xB = 0, yB = blockIdx.y * BLOCK_M;
    int xC = xA + tx * THREAD_N, yC = yB + ty * THREAD_M;

    for (int i = 0; i < N; i += BLOCK_K, yA += BLOCK_K, xB += BLOCK_K) {
        // Copy data to sA first
        // sA should be BLOCK_N x BLOCK_K
        y = threadIdx.x % BLOCK_K;
        #pragma unroll
        for (int j = 0; j < (BLOCK_N * BLOCK_K + GRID_SIZE - 1) / GRID_SIZE; ++j) {
            x = j * GRID_SIZE / BLOCK_K + threadIdx.x / BLOCK_K;
            // padding 0s & avoid branch
            sA[x*BLOCK_K + y] = -(~((N - xA - x - 1) >> 31) * A[(xA + x) * N + yA + y]);
        }

        // now for sB
        // sB should be BLOCK_K x BLOCK_M
        y = threadIdx.x % BLOCK_M;
        #pragma unroll
        for (int j = 0; j < (BLOCK_M * BLOCK_K + GRID_SIZE - 1) / GRID_SIZE; ++j) {
            x = threadIdx.x / BLOCK_M + j * GRID_SIZE / BLOCK_M;
            sB[x*BLOCK_M + y] = -(~((N - xB - x - 1) >> 31) * B[(xB + x) * N + yB + y]);
        }
        __syncthreads();

        // Micro-Kernel
        #pragma unroll
        for (int j = 0; j < BLOCK_K; ++j) {
            #pragma unroll
            for (int k = 0; k < THREAD_N; ++k) {
                rA[k] = sA[(k + THREAD_N * tx)*BLOCK_K + j];
            }
            #pragma unroll
            for (int k = 0; k < THREAD_M; ++k) {
                rB[k] = sB[j*BLOCK_M + k + THREAD_M * ty];
            }
            #pragma unroll
            for (int px = 0; px < THREAD_N; ++px) {
                #pragma unroll
                for (int py = 0; py < THREAD_M; ++py) {
                    c[px][py] += rA[px] * rB[py];
                    // printf("CUR: %f", c[px][py]);
                }
            }
        }
        __syncthreads(); // ensure next loop's data is loaded
    }

    // Copy result to C
    #pragma unroll
    for (int px = 0; xC + px < N && px < THREAD_N; ++px) {
        #pragma unroll
        for (int py = 0; yC + py < N && py < THREAD_M; ++py) {
            // c[px][py] = 0;
            C[(xC + px) * N + yC + py] = c[px][py];
        }
    }
}
#endif
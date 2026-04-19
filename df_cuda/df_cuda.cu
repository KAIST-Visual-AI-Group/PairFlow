#include <torch/extension.h>
#include <cstdio>
#include <math.h>
#include "df_cuda.h"

#if defined(__CUDA_ARCH__)
    #define POPCOUNT64(x) __popcll(x)
#elif defined(__GNUC__) || defined(__clang__)
    #define POPCOUNT64(x) __builtin_popcountll(x)
#else
    #define POPCOUNT64(x) __popcnt64(x)
#endif






#define TILE_SIZE  16
#define TILE_SIZE7 128
#define TILE_SIZE2 256
#define LOAD_SIZE1 8
#define TILE2_LOAD1 (TILE_SIZE2 / LOAD_SIZE1)

__global__ void cumulate_bits(
    const uint64_t* __restrict__ data, // (N, D)
    uint64_t* out, // (N, bD)
    const uint64_t N,
    const uint64_t D,
    const uint64_t num_bits
){
    uint64_t i1 = blockIdx.x * TILE_SIZE2, i0 = threadIdx.x;
    uint64_t l1 = threadIdx.x / LOAD_SIZE1, l0 = threadIdx.x % LOAD_SIZE1;
    uint64_t _64_bV = 64 / num_bits, bD = (D - 1) / _64_bV + 1;
    
    __shared__ uint64_t cache[TILE_SIZE2][LOAD_SIZE1];
    __shared__ uint64_t out_cache[TILE_SIZE2][LOAD_SIZE1];
    
    for (uint64_t d1 = 0; d1 < bD; d1 += LOAD_SIZE1) {
        for (uint64_t d0 = 0; d0 < LOAD_SIZE1; d0++) {
            // out: i1 + i0, d1 * LOAD_SIZE1 + d0
            // data: (d1 * LOAD_SIZE1 + d0) * _64_bV ~ + _64bV

            uint64_t out_val = 0ULL;
            uint64_t j_base = (d1 + d0) * _64_bV;

            for (uint64_t j0 = 0; j0 < _64_bV; j0 += LOAD_SIZE1) {
                for (uint64_t j1 = 0; j1 < TILE_SIZE2; j1 += TILE2_LOAD1)
                    cache[j1 + l1][l0] = i1 + j1 + l1 < N && j0 + l0 < _64_bV? data[(i1 + j1 + l1) * D + j_base + j0 + l0]: 0ULL;
                __syncthreads();

                for (uint64_t j1 = 0; j1 < LOAD_SIZE1 && j0 + j1 < _64_bV; j1++)
                    out_val |= cache[i0][j1] << (num_bits * (j0 + j1));
                __syncthreads();
            }

            out_cache[i0][d0] = out_val;
            __syncthreads();
        }
        
        for (uint64_t d0 = 0; d0 < TILE_SIZE2; d0 += TILE2_LOAD1)
            if (i1 + d0 + l1 < N && d1 + l0 < bD)
                out[(i1 + d0 + l1) * bD + d1 + l0] = out_cache[d0 + l1][l0];
        __syncthreads();
    }
}


void fill_bits_(
    const uint64_t* data,
    uint64_t* out,
    uint64_t N,
    uint64_t D,
    uint64_t num_bits
){
    dim3 grid((N - 1) / TILE_SIZE2 + 1, 1, 1);
    cumulate_bits <<< grid, TILE_SIZE2 >>> (data, out, N, D, num_bits);
}


void fill_bits(
    torch::Tensor& data,
    torch::Tensor& out,
    uint64_t num_bits
){
    uint64_t N = data.size(0), D = data.size(1);
    fill_bits_(
        data.data_ptr<uint64_t>(),
        out.data_ptr<uint64_t>(),
        N, D, num_bits
    );
}





__global__ void count_same_tokens_cuda(
    const uint64_t* __restrict__ x0, // (B, bD)
    const uint64_t* __restrict__ x1, // (N, bD)
    int* out,
    const uint64_t B,
    const uint64_t N,
    const uint64_t D,
    const uint64_t num_bits
){
    uint64_t _64_bV = 64 / num_bits, bD = (D - 1) / _64_bV + 1,
             masko = (1ULL << (_64_bV * num_bits)) - 1,
             maskl = D % _64_bV? (1ULL << ((D % _64_bV) * num_bits)) - 1: masko,
             lo_mask = masko / ((1ULL << num_bits) - 1),
             hi_mask = lo_mask << (num_bits - 1),
             i = blockIdx.x * TILE_SIZE7,  j = blockIdx.y * TILE_SIZE7,
             i0 = threadIdx.x / TILE_SIZE, j0 = threadIdx.x % TILE_SIZE;

    int32_t out_vals[LOAD_SIZE1][LOAD_SIZE1] = {0};
    uint64_t x0_cache2[TILE_SIZE] = {0};

    __shared__ uint64_t x0_cache[TILE_SIZE7][TILE_SIZE];
    __shared__ uint64_t x1_cache[TILE_SIZE7][TILE_SIZE];
    
    for (uint64_t d1 = 0; d1 < bD; d1 += TILE_SIZE) {
        for (uint64_t k = 0; k < TILE_SIZE7; k += TILE_SIZE)
            x0_cache[k + i0][j0] = i + k + i0 < B && d1 + j0 < bD? x0[(i + k + i0) * bD + d1 + j0]: 0ULL;
        for (uint64_t k = 0; k < TILE_SIZE7; k += TILE_SIZE)
            x1_cache[k + i0][j0] = j + k + i0 < N && d1 + j0 < bD? x1[(j + k + i0) * bD + d1 + j0]: 0ULL;
        __syncthreads();

        for (uint64_t im = 0; i + im < B && im < TILE_SIZE7; im += TILE_SIZE) {
            for (uint64_t d0 = 0; d0 + d1 < bD && d0 < TILE_SIZE; d0++) x0_cache2[d0] = x0_cache[im + i0][d0];

            for (uint64_t jm = 0; j + jm < N && jm < TILE_SIZE7; jm += TILE_SIZE) {
                int32_t out_val = 0;
                for (uint64_t d0 = 0; d0 + d1 < bD && d0 < TILE_SIZE; d0++) {
                    uint64_t z = (x0_cache2[d0] ^ ~x1_cache[jm + j0][d0]) & (d1 + d0 == bD - 1? maskl: masko);
                    
                    if (num_bits == 1)
                        out_val += POPCOUNT64(z);
                    else
                        out_val += POPCOUNT64(((z & ~hi_mask) + lo_mask) & z & hi_mask);
                }
                if (i + im + i0 < B && j + jm + j0 < N)
                    out_vals[im / TILE_SIZE][jm / TILE_SIZE] += out_val;
            }
        }
        __syncthreads();
    }

    for (uint64_t im = 0; im < TILE_SIZE7; im += TILE_SIZE) {
        for (uint64_t jm = 0; jm < TILE_SIZE7; jm += TILE_SIZE) {
            if (i + im + i0 < B && j + jm + j0 < N)
                out[(i + im + i0) * N + j + jm + j0] = out_vals[im / TILE_SIZE][jm / TILE_SIZE];
        }
    }
}


void count_same_tokens_(
    const uint64_t* x0,
    const uint64_t* x1,
    int* out,
    uint64_t B,
    uint64_t N,
    uint64_t D,
    uint64_t num_bits
){
    dim3 grid((B - 1) / TILE_SIZE7 + 1, (N - 1) / TILE_SIZE7 + 1, 1);
    count_same_tokens_cuda <<< grid, TILE_SIZE2 >>> (x0, x1, out, B, N, D, num_bits);
}


void count_same_tokens(
    torch::Tensor& x0,
    torch::Tensor& x1,
    torch::Tensor& out,
    uint64_t D,
    uint64_t num_bits
){
    uint64_t B = x0.size(0), N = x1.size(0);
    count_same_tokens_(
        x0.data_ptr<uint64_t>(),
        x1.data_ptr<uint64_t>(),
        out.data_ptr<int>(),
        B, N, D, num_bits
    );
}










// __global__ void sum_same_tokens_cuda(
//     const uint64_t* __restrict__ x0, // (B, bD)
//     const uint64_t* __restrict__ x1, // (N, bD)
//     const float* __restrict__ val, // (B, N)
//     float* out,
//     const uint64_t B,
//     const uint64_t N,
//     const uint64_t D,
//     const uint64_t num_bits
// ){
//     uint64_t _64_bV = 64 / num_bits, bD = (D - 1) / _64_bV + 1,
//              i = blockIdx.x * TILE_SIZE7,  j = blockIdx.y * TILE_SIZE7,
//              i0 = threadIdx.x / TILE_SIZE, j0 = threadIdx.x % TILE_SIZE,
//              ds = j / _64_bV, de = (j + TILE_SIZE7 - 1) / _64_bV + 1;

//     uint32_t mask = (1U << num_bits) - 1;
//     float out_vals[LOAD_SIZE1][LOAD_SIZE1] = {0.0f};
//     uint32_t x0_cache[LOAD_SIZE1][LOAD_SIZE1] = {0};
//     uint32_t x1_cache2[TILE_SIZE] = {0};

//     for (uint64_t jm = 0; j + jm < D && jm < TILE_SIZE7; jm += TILE_SIZE) {
//         uint64_t jc = j + jm + j0, jc0 = jc / _64_bV, jc1 = jc % _64_bV;
//         for (uint64_t im = 0; i + im < B && im < TILE_SIZE7; im += TILE_SIZE) {
//             x0_cache[im / TILE_SIZE][jm / TILE_SIZE] = (uint32_t)(x0[(i + im + i0) * bD + jc0] >> (jc1 * num_bits)) & mask;
//         }
//     }

//     __shared__ float val_cache[TILE_SIZE7][TILE_SIZE];
//     __shared__ uint64_t x1_cache[TILE_SIZE][TILE_SIZE7 / 2 + 1];
    
//     for (uint64_t n1 = 0; n1 < N; n1 += TILE_SIZE) {
//         for (uint64_t k = 0; k < TILE_SIZE7; k += TILE_SIZE)
//             val_cache[k + i0][j0] = i + k + i0 < B && n1 + j0 < N? val[(i + k + i0) * N + n1 + j0]: 0.0f;
//         for (uint64_t k = 0; ds + k < de; k += TILE_SIZE)
//             x1_cache[i0][k + j0] = j + k + j0 < N && ds + k + j0 < de? x1[(n1 + i0) * bD + ds + k + j0]: 0ULL;
//         __syncthreads();
        
//         for (uint64_t jm = 0; j + jm < D && jm < TILE_SIZE7; jm += TILE_SIZE) {
//             uint64_t jc = j + jm + j0, jc0 = jc / _64_bV, jc1 = jc % _64_bV;
//             for (uint64_t n0 = 0; n0 + n1 < N && n0 < TILE_SIZE; n0++) {
//                 x1_cache2[n0] = (uint32_t)(x1_cache[n0][jc0 - ds] >> (jc1 * num_bits));
//             }

//             for (uint64_t im = 0; i + im < B && im < TILE_SIZE7; im += TILE_SIZE) {
//                 float out_val = 0;
//                 for (uint64_t n0 = 0; n0 + n1 < N && n0 < TILE_SIZE; n0++) {
//                     if (mask == (mask & (x0_cache[im / TILE_SIZE][jm / TILE_SIZE] ^ ~x1_cache2[n0])))
//                         out_val += val_cache[im + i0][n0];
//                 }
//                 if (i + im + i0 < B && j + jm + j0 < D)
//                     out_vals[im / TILE_SIZE][jm / TILE_SIZE] += out_val;
//             }
//         }
//         __syncthreads();
//     }

//     for (uint64_t im = 0; im < TILE_SIZE7; im += TILE_SIZE) {
//         for (uint64_t jm = 0; jm < TILE_SIZE7; jm += TILE_SIZE) {
//             if (i + im + i0 < B && j + jm + j0 < D)
//                 out[(i + im + i0) * D + j + jm + j0] = out_vals[im / TILE_SIZE][jm / TILE_SIZE];
//         }
//     }
// }





// void sum_same_tokens_(
//     const uint64_t* x0,
//     const uint64_t* x1,
//     const float* val,
//     float* out,
//     uint64_t B,
//     uint64_t N,
//     uint64_t D,
//     uint64_t num_bits
// ){
//     dim3 grid((B - 1) / TILE_SIZE7 + 1, (D - 1) / TILE_SIZE7 + 1, 1);
//     sum_same_tokens_cuda <<< grid, TILE_SIZE2 >>> (x0, x1, val, out, B, N, D, num_bits);
// }


// void sum_same_tokens(
//     torch::Tensor& x0,
//     torch::Tensor& x1,
//     torch::Tensor& val,
//     torch::Tensor& out,
//     uint64_t D,
//     uint64_t num_bits
// ){
//     uint64_t B = x0.size(0), N = x1.size(0);
//     sum_same_tokens_(
//         x0.data_ptr<uint64_t>(),
//         x1.data_ptr<uint64_t>(),
//         val.data_ptr<float>(),
//         out.data_ptr<float>(),
//         B, N, D, num_bits
//     );
// }








__global__ void sum_same_tokens_large_cuda(
    const uint64_t* __restrict__ x0, // (B, bD)
    const uint64_t* __restrict__ x1, // (N, bD)
    const float* __restrict__ val, // (B, N)
    float* out,
    const uint64_t B,
    const uint64_t N,
    const uint64_t D,
    const uint64_t num_bits
){
    uint64_t _64_bV = 64 / num_bits, bD = (D - 1) / _64_bV + 1,
             i = blockIdx.x * TILE_SIZE7,  j = blockIdx.y * TILE_SIZE7, n = blockIdx.z * TILE_SIZE2,
             i0 = threadIdx.x / TILE_SIZE, j0 = threadIdx.x % TILE_SIZE,
             ds = j / _64_bV, de = (j + TILE_SIZE7 - 1) / _64_bV + 1;

    uint32_t mask = (1U << num_bits) - 1;
    float out_vals[LOAD_SIZE1][LOAD_SIZE1] = {0.0f};
    uint32_t x0_cache[LOAD_SIZE1][LOAD_SIZE1] = {0};
    uint32_t x1_cache2[TILE_SIZE] = {0};

    for (uint64_t jm = 0; j + jm < D && jm < TILE_SIZE7; jm += TILE_SIZE) {
        uint64_t jc = j + jm + j0, jc0 = jc / _64_bV, jc1 = jc % _64_bV;
        for (uint64_t im = 0; i + im < B && im < TILE_SIZE7; im += TILE_SIZE) {
            x0_cache[im / TILE_SIZE][jm / TILE_SIZE] = (uint32_t)(x0[(i + im + i0) * bD + jc0] >> (jc1 * num_bits)) & mask;
        }
    }

    __shared__ float val_cache[TILE_SIZE7][TILE_SIZE];
    __shared__ uint64_t x1_cache[TILE_SIZE][TILE_SIZE7 / 2 + 1];
    
    for (uint64_t n1 = 0; n + n1 < N && n1 < TILE_SIZE2; n1 += TILE_SIZE) {
        for (uint64_t k = 0; k < TILE_SIZE7; k += TILE_SIZE)
            val_cache[k + i0][j0] = i + k + i0 < B && n + n1 + j0 < N? val[(i + k + i0) * N + n + n1 + j0]: 0.0f;
        for (uint64_t k = 0; ds + k < de; k += TILE_SIZE)
            x1_cache[i0][k + j0] = n + n1 + i0 < N && ds + k + j0 < de? x1[(n + n1 + i0) * bD + ds + k + j0]: 0ULL;
        __syncthreads();
        
        for (uint64_t jm = 0; j + jm < D && jm < TILE_SIZE7; jm += TILE_SIZE) {
            uint64_t jc = j + jm + j0, jc0 = jc / _64_bV, jc1 = jc % _64_bV;
            for (uint64_t n0 = 0; n + n1 + n0 < N && n0 < TILE_SIZE; n0++) {
                x1_cache2[n0] = (uint32_t)(x1_cache[n0][jc0 - ds] >> (jc1 * num_bits));
            }

            for (uint64_t im = 0; i + im < B && im < TILE_SIZE7; im += TILE_SIZE) {
                float out_val = 0;
                for (uint64_t n0 = 0; n + n1 + n0 < N && n0 < TILE_SIZE; n0++) {
                    if (mask == (mask & (x0_cache[im / TILE_SIZE][jm / TILE_SIZE] ^ ~x1_cache2[n0])))
                        out_val += val_cache[im + i0][n0];
                }
                if (i + im + i0 < B && j + jm + j0 < D)
                    out_vals[im / TILE_SIZE][jm / TILE_SIZE] += out_val;
            }
        }
        __syncthreads();
    }

    for (uint64_t im = 0; im < TILE_SIZE7; im += TILE_SIZE) {
        for (uint64_t jm = 0; jm < TILE_SIZE7; jm += TILE_SIZE) {
            if (i + im + i0 < B && j + jm + j0 < D)
                atomicAdd(&out[(i + im + i0) * D + j + jm + j0], out_vals[im / TILE_SIZE][jm / TILE_SIZE]);
        }
    }
}





void sum_same_tokens_large_(
    const uint64_t* x0,
    const uint64_t* x1,
    const float* val,
    float* out,
    uint64_t B,
    uint64_t N,
    uint64_t D,
    uint64_t num_bits
){
    dim3 grid((B - 1) / TILE_SIZE7 + 1, (D - 1) / TILE_SIZE7 + 1, (N - 1) / TILE_SIZE2 + 1);
    sum_same_tokens_large_cuda <<< grid, TILE_SIZE2 >>> (x0, x1, val, out, B, N, D, num_bits);
}


void sum_same_tokens_large(
    torch::Tensor& x0,
    torch::Tensor& x1,
    torch::Tensor& val,
    torch::Tensor& out,
    uint64_t D,
    uint64_t num_bits
){
    uint64_t B = x0.size(0), N = x1.size(0);
    sum_same_tokens_large_(
        x0.data_ptr<uint64_t>(),
        x1.data_ptr<uint64_t>(),
        val.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, D, num_bits
    );
}










__global__ void calculate_val_tn1_cuda(
    const int* __restrict__ num, // (B, N)
    const int* __restrict__ max_nums, // (B)
    float* out, // (B, N)
    const uint64_t B,
    const uint64_t N,
    const float gamma
){
    uint64_t i = blockIdx.x * TILE_SIZE7,  j = blockIdx.y * TILE_SIZE7,
             i0 = threadIdx.x / TILE_SIZE, j0 = threadIdx.x % TILE_SIZE, k = threadIdx.x;
    
    __shared__ int32_t max_cache[TILE_SIZE7];
    if (k < TILE_SIZE7) max_cache[k] = i + k < B? max_nums[i + k]: 0;
    __syncthreads();

    for (uint64_t i1 = 0; i1 < TILE_SIZE7; i1 += TILE_SIZE) {
        for (uint64_t j1 = 0; j1 < TILE_SIZE7; j1 += TILE_SIZE) {
            uint64_t in = i + i1 + i0, jn = j + j1 + j0;
            if (in < B && jn < N)
                out[in * N + jn] = powf(gamma, (float)(num[in * N + jn] - max_cache[i1 + i0]));
        }
    }
}


void calculate_val_tn1_(
    const int* num,
    const int* max_nums,
    float* out,
    uint64_t B,
    uint64_t N,
    float gamma
){
    dim3 grid((B - 1) / TILE_SIZE7 + 1, (N - 1) / TILE_SIZE7 + 1, 1);
    calculate_val_tn1_cuda <<< grid, TILE_SIZE2 >>> (num, max_nums, out, B, N, gamma);
}


void calculate_val_tn1(
    torch::Tensor& num,
    torch::Tensor& max_nums,
    torch::Tensor& out,
    float gamma
){
    uint64_t B = num.size(0), N = num.size(1);
    calculate_val_tn1_(
        num.data_ptr<int>(),
        max_nums.data_ptr<int>(),
        out.data_ptr<float>(),
        B, N, gamma
    );
}









__global__ void calculate_val_te1_cuda(
    const int* __restrict__ num, // (B, N)
    const int* __restrict__ max_nums, // (B)
    float* out, // (B, N)
    const uint64_t B,
    const uint64_t N
){
    uint64_t i = blockIdx.x * TILE_SIZE7,  j = blockIdx.y * TILE_SIZE7,
             i0 = threadIdx.x / TILE_SIZE, j0 = threadIdx.x % TILE_SIZE, k = threadIdx.x;
    
    __shared__ int32_t max_cache[TILE_SIZE7];
    if (k < TILE_SIZE7) max_cache[k] = i + k < B? max_nums[i + k]: 0;
    __syncthreads();

    for (uint64_t i1 = 0; i1 < TILE_SIZE7; i1 += TILE_SIZE) {
        for (uint64_t j1 = 0; j1 < TILE_SIZE7; j1 += TILE_SIZE) {
            uint64_t in = i + i1 + i0, jn = j + j1 + j0;
            if (in < B && jn < N)
                out[in * N + jn] = num[in * N + jn] == max_cache[i1 + i0]? 1.0f: 0.0f;
        }
    }
}


void calculate_val_te1_(
    const int* num,
    const int* max_nums,
    float* out,
    uint64_t B,
    uint64_t N
){
    dim3 grid((B - 1) / TILE_SIZE7 + 1, (N - 1) / TILE_SIZE7 + 1, 1);
    calculate_val_te1_cuda <<< grid, TILE_SIZE2 >>> (num, max_nums, out, B, N);
}


void calculate_val_te1(
    torch::Tensor& num,
    torch::Tensor& max_nums,
    torch::Tensor& out
){
    uint64_t B = num.size(0), N = num.size(1);
    calculate_val_te1_(
        num.data_ptr<int>(),
        max_nums.data_ptr<int>(),
        out.data_ptr<float>(),
        B, N
    );
}










__global__ void normalize_cuda(
    float* val, // (B, N)
    const float* __restrict__ sum_vals, // (B)
    const uint64_t B,
    const uint64_t N
){
    uint64_t i = blockIdx.x * TILE_SIZE7,  j = blockIdx.y * TILE_SIZE7,
             i0 = threadIdx.x / TILE_SIZE, j0 = threadIdx.x % TILE_SIZE, k = threadIdx.x;
    
    __shared__ float sum_cache[TILE_SIZE7];
    if (k < TILE_SIZE7) sum_cache[k] = i + k < B? sum_vals[i + k]: 1.0f;
    __syncthreads();

    for (uint64_t i1 = 0; i1 < TILE_SIZE7; i1 += TILE_SIZE) {
        for (uint64_t j1 = 0; j1 < TILE_SIZE7; j1 += TILE_SIZE) {
            if (i + i1 + i0 < B && j + j1 + j0 < N)
                val[(i + i1 + i0) * N + j + j1 + j0] /= sum_cache[i1 + i0];
        }
    }
}


void normalize_(
    float* val,
    const float* sum_vals,
    uint64_t B,
    uint64_t N
){
    dim3 grid((B - 1) / TILE_SIZE7 + 1, (N - 1) / TILE_SIZE7 + 1, 1);
    normalize_cuda <<< grid, TILE_SIZE2 >>> (val, sum_vals, B, N);
}


void normalize(
    torch::Tensor& val,
    torch::Tensor& sum_vals
){
    uint64_t B = val.size(0), N = val.size(1);
    normalize_(
        val.data_ptr<float>(),
        sum_vals.data_ptr<float>(),
        B, N
    );
}









__global__ void get_thres_cuda(
    float* val, // (B, D)
    const uint64_t B,
    const uint64_t D,
    const int V,
    const float tau,
    const float t,
    const float h
){
    uint64_t i = blockIdx.x * TILE_SIZE,   j = blockIdx.y * TILE_SIZE,
             i0 = threadIdx.x / TILE_SIZE, j0 = threadIdx.x % TILE_SIZE;
    
    if (i + i0 < B && j + j0 < D){
        float v = val[(i + i0) * D + j + j0], Vf = (float)V, f;
        f = 1.0f + (Vf - 1.0f) * expf(logf(h * v / (1.0f - t + t * Vf - h * (Vf - 1.0f) * v)) / tau);
        val[(i + i0) * D + j + j0] = 1.0 / f;
    }
}


void get_thres_(
    float* val,
    uint64_t B,
    uint64_t D,
    int V,
    float tau,
    float t,
    float h
){
    dim3 grid((B - 1) / TILE_SIZE + 1, (D - 1) / TILE_SIZE + 1, 1);
    get_thres_cuda <<< grid, TILE_SIZE2 >>> (val, B, D, V, tau, t, h);
}


void get_thres(
    torch::Tensor& val,
    int V,
    float tau,
    float t,
    float h
){
    uint64_t B = val.size(0), D = val.size(1);
    get_thres_(
        val.data_ptr<float>(),
        B, D, V, tau, t, h
    );
}
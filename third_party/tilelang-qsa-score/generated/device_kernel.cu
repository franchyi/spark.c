#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const int* __restrict__ ContextLens, const bfloat16_t* __restrict__ KCache, float* __restrict__ Logits, const int* __restrict__ PageTable, const bfloat16_t* __restrict__ Q, float Scale, int batch, int max_model_len, int max_pages, int pages);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const int* __restrict__ ContextLens, const bfloat16_t* __restrict__ KCache, float* __restrict__ Logits, const int* __restrict__ PageTable, const bfloat16_t* __restrict__ Q, float Scale, int batch, int max_model_len, int max_pages, int pages) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 2048));
  float scores[4];
  float reduced[2];
  *(uint4*)(((bfloat16_t*)q_shared) + (((((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
  int context_len = ContextLens[((int64_t)((int)blockIdx.x))];
  if ((((int)blockIdx.y) * 64) < context_len) {
    for (int sp = 0; sp < 4; ++sp) {
      if (((((int)blockIdx.y) * 64) + (sp * 16)) < context_len) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          int condval_1;
          if ((((((int)blockIdx.y) * 4) + sp) < max_pages)) {
            condval_1 = PageTable[(((((int64_t)((int)blockIdx.y)) * (int64_t)4) + (((int64_t)((int)blockIdx.x)) * ((int64_t)max_pages))) + ((int64_t)sp))];
          } else {
            condval_1 = 0;
          }
          int condval_2;
          if ((((((int)blockIdx.y) * 4) + sp) < max_pages)) {
            condval_2 = PageTable[(((((int64_t)((int)blockIdx.y)) * (int64_t)4) + (((int64_t)((int)blockIdx.x)) * ((int64_t)max_pages))) + ((int64_t)sp))];
          } else {
            condval_2 = 0;
          }
          uint4 condval;
          if (((0 <= condval_1) && (condval_2 < pages))) {
            int64_t condval_3;
            if ((((((int64_t)((int)blockIdx.y)) * (int64_t)4) + ((int64_t)sp)) < ((int64_t)max_pages))) {
              condval_3 = ((int64_t)PageTable[(((((int64_t)((int)blockIdx.y)) * (int64_t)4) + (((int64_t)((int)blockIdx.x)) * ((int64_t)max_pages))) + ((int64_t)sp))]);
            } else {
              condval_3 = (int64_t)0;
            }
            int64_t condval_4;
            if ((((((int64_t)((int)blockIdx.y)) * (int64_t)4) + ((int64_t)sp)) < ((int64_t)max_pages))) {
              condval_4 = ((int64_t)PageTable[(((((int64_t)((int)blockIdx.y)) * (int64_t)4) + (((int64_t)((int)blockIdx.x)) * ((int64_t)max_pages))) + ((int64_t)sp))]);
            } else {
              condval_4 = (int64_t)0;
            }
            condval = *(uint4*)(KCache + (((condval_4 * (int64_t)2048) + (((int64_t)i) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
          } else {
            condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
          }
          *(uint4*)(((bfloat16_t*)k_shared) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (sp * 1024)) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
        }
      }
    }
  }
  __syncthreads();
  if ((((int)blockIdx.y) * 64) < context_len) {
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[4];
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_shared)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x2((&(((bfloat16_t*)q_shared)[((((((ki >> 2) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      }
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      scores[i_1] = max(scores[i_1], 0x0p+0f/*0.000000e+00*/);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      reduced[i_2] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 2; ++rv) {
        reduced[i_2] = (reduced[i_2] + scores[((i_2 * 2) + rv)]);
      }
      reduced[i_2] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(reduced[i_2]);
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        if (((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < context_len) {
          if (((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < max_model_len) {
            Logits[(((((((int64_t)((int)blockIdx.y)) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((int64_t)i_3) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) + (((int64_t)((int)blockIdx.x)) * ((int64_t)max_model_len)))] = (reduced[i_3] / Scale);
          }
        }
      }
    }
  }
}


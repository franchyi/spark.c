/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Indexed pointer preparation derived from FlashInfer's pinned SM120 grouped
 * NVFP4 template at 906181e3f4cf4bcc81835fb480db4011bbd80b62. The CUTLASS
 * mainloop and epilogue are unchanged; only B/SFB address selection differs so
 * a compact group can select a logical expert in a strided resident bank.
 */

#include "internal/nvfp4_grouped_backend.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <type_traits>

#include "flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh"

namespace flashinfer {
namespace group_gemm {

using namespace cute;

template <int ScaleGranularity, typename ScaleConfig, typename ElementA,
          typename ElementB, typename ElementSFA, typename ElementSFB,
          typename ElementD, typename ProblemShape, typename StrideA,
          typename StrideB, typename StrideD, typename LayoutSFA,
          typename LayoutSFB>
__global__ void compute_sm120_indexed_nvfp4_group_gemm_args(
    ElementA* A, ElementB* B, ElementSFA* SFA, ElementSFB* SFB, ElementD* D,
    int* m_indptr, const int32_t* logical_group_ids,
    uint32_t source_group_count, uint64_t packed_group_stride_bytes,
    uint64_t scale_group_stride_bytes, int n, int k, int num_groups,
    ProblemShape* problem_sizes, const ElementA** A_ptr,
    const ElementB** B_ptr, const ElementSFA** SFA_ptr,
    const ElementSFB** SFB_ptr, ElementD** D_ptr, StrideA* stride_A,
    StrideB* stride_B, StrideD* stride_D, LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB) {
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= num_groups) return;

  constexpr size_t kSwizzledMnAlignment = 128;
  constexpr size_t kSwizzledKAlignment =
      static_cast<size_t>(ScaleGranularity) * 4;
  const size_t sf_n =
      (static_cast<size_t>(n) + kSwizzledMnAlignment - 1) /
      kSwizzledMnAlignment * kSwizzledMnAlignment;
  const size_t swizzled_k =
      (static_cast<size_t>(k) + kSwizzledKAlignment - 1) /
      kSwizzledKAlignment * kSwizzledKAlignment;
  const size_t sf_k = swizzled_k / static_cast<size_t>(ScaleGranularity);
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
     (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  const int m_offset = m_indptr[group];
  const int m_offset_next = m_indptr[group + 1];
  const size_t m = static_cast<size_t>(m_offset_next - m_offset);
  const size_t sf_m_offset =
      (static_cast<size_t>(m_offset) + static_cast<size_t>(group) *
                                           (kSwizzledMnAlignment - 1)) /
      kSwizzledMnAlignment * kSwizzledMnAlignment;

  // The host mirror is range/uniqueness validated before launch. Retain this
  // device guard so a stale or mismatched device mirror cannot form an OOB
  // resident-bank address.
  int32_t logical_group = logical_group_ids[group];
  if (logical_group < 0 ||
      static_cast<uint32_t>(logical_group) >= source_group_count) {
    logical_group = 0;
  }

  problem_sizes[group] = ProblemShape(m, n, k);
  stride_A[group] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  stride_B[group] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  stride_D[group] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  // ElementA/B are sub-byte types. Byte addressing avoids element-unit
  // ambiguity while preserving the 16-byte alignment enforced by the ABI.
  A_ptr[group] = reinterpret_cast<const ElementA*>(
      reinterpret_cast<const uint8_t*>(A) +
      static_cast<size_t>(m_offset) * static_cast<size_t>(k) / 2);
  B_ptr[group] = reinterpret_cast<const ElementB*>(
      reinterpret_cast<const uint8_t*>(B) +
      static_cast<uint64_t>(logical_group) * packed_group_stride_bytes);
  D_ptr[group] = reinterpret_cast<ElementD*>(
      reinterpret_cast<uint8_t*>(D) +
      static_cast<size_t>(m_offset) * static_cast<size_t>(n) *
          sizeof(ElementD));

  layout_SFA[group] = ScaleConfig::tile_atom_to_shape_SFA(
      make_shape(static_cast<int>(m), static_cast<int>(sf_n),
                 static_cast<int>(swizzled_k), 1));
  SFA_ptr[group] = reinterpret_cast<const ElementSFA*>(
      reinterpret_cast<const uint8_t*>(SFA) + sf_m_offset * sf_k);
  layout_SFB[group] = ScaleConfig::tile_atom_to_shape_SFB(
      make_shape(static_cast<int>(m), static_cast<int>(sf_n),
                 static_cast<int>(swizzled_k), 1));
  SFB_ptr[group] = reinterpret_cast<const ElementSFB*>(
      reinterpret_cast<const uint8_t*>(SFB) +
      static_cast<uint64_t>(logical_group) * scale_group_stride_bytes);
}

cudaError_t CutlassIndexedNVFP4GroupwiseScaledGroupGEMMSM120(
    void* int_buffer, size_t int_buffer_size_in_bytes, void* float_buffer,
    size_t float_buffer_size_in_bytes, cutlass::float_e2m1_t* A,
    cutlass::float_e2m1_t* B, cutlass::float_ue4m3_t* SFA,
    cutlass::float_ue4m3_t* SFB, cutlass::bfloat16_t* D, float* alpha,
    int* m_indptr, const int32_t* logical_group_ids,
    uint32_t source_group_count, uint64_t packed_group_stride_bytes,
    uint64_t scale_group_stride_bytes, int n, int k, int num_groups,
    cudaStream_t stream, int device_id) {
  if (num_groups == 0) return cudaSuccess;

  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementC = void;
  using LayoutC = void;
  using ElementAccumulator = float;
  using ElementCompute = float;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  constexpr int AlignmentC = 0;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  FLASHINFER_CHECK(k % std::max(AlignmentA, AlignmentB) == 0,
                   "k violates grouped NVFP4 alignment");
  FLASHINFER_CHECK(n % AlignmentD == 0,
                   "n violates grouped NVFP4 alignment");

  using ElementAMainloop = cutlass::nv_float4_t<ElementA>;
  using ElementBMainloop = cutlass::nv_float4_t<ElementB>;
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ClusterShape = Shape<_1, _1, _1>;
  using ThreadBlockShape = Shape<Int<128>, Int<128>, Int<256>>;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
          ThreadBlockShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutD*, AlignmentD,
          ElementD, LayoutD*, AlignmentD, EpilogueSchedule>::CollectiveOp;
  using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
          ElementAMainloop, LayoutA*, AlignmentA, ElementBMainloop, LayoutB*,
          AlignmentB, ElementAccumulator, ThreadBlockShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using ScaleConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using LayoutSFA =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  constexpr int ScaleGranularity =
      Gemm::GemmKernel::CollectiveMainloop::TiledMma::SFVecSize;
  static_assert(ScaleGranularity == 16,
                "Scale granularity must remain NVFP4 group-size 16");

  AlignedAllocator allocator(int_buffer, int_buffer_size_in_bytes);
  auto problem_sizes =
      allocator.aligned_alloc<typename ProblemShape::UnderlyingProblemShape>(
          num_groups * sizeof(typename ProblemShape::UnderlyingProblemShape),
          16, "sm120_indexed_group_gemm_problem_sizes");
  auto A_ptr = allocator.aligned_alloc<const typename Gemm::ElementA*>(
      num_groups * sizeof(const typename Gemm::ElementA*), 16,
      "sm120_indexed_group_gemm_A_ptr");
  auto B_ptr = allocator.aligned_alloc<const typename Gemm::ElementB*>(
      num_groups * sizeof(const typename Gemm::ElementB*), 16,
      "sm120_indexed_group_gemm_B_ptr");
  auto D_ptr =
      allocator.aligned_alloc<typename Gemm::EpilogueOutputOp::ElementOutput*>(
          num_groups *
              sizeof(typename Gemm::EpilogueOutputOp::ElementOutput*),
          16, "sm120_indexed_group_gemm_D_ptr");
  auto SFA_ptr = allocator.aligned_alloc<const ElementSFA*>(
      num_groups * sizeof(const ElementSFA*), 16,
      "sm120_indexed_group_gemm_SFA_ptr");
  auto SFB_ptr = allocator.aligned_alloc<const ElementSFB*>(
      num_groups * sizeof(const ElementSFB*), 16,
      "sm120_indexed_group_gemm_SFB_ptr");
  auto stride_A = allocator.aligned_alloc<StrideA>(
      num_groups * sizeof(StrideA), 16, "sm120_indexed_group_gemm_stride_A");
  auto stride_B = allocator.aligned_alloc<StrideB>(
      num_groups * sizeof(StrideB), 16, "sm120_indexed_group_gemm_stride_B");
  auto stride_D = allocator.aligned_alloc<StrideD>(
      num_groups * sizeof(StrideD), 16, "sm120_indexed_group_gemm_stride_D");
  auto layout_SFA = allocator.aligned_alloc<LayoutSFA>(
      num_groups * sizeof(LayoutSFA), 16,
      "sm120_indexed_group_gemm_layout_SFA");
  auto layout_SFB = allocator.aligned_alloc<LayoutSFB>(
      num_groups * sizeof(LayoutSFB), 16,
      "sm120_indexed_group_gemm_layout_SFB");

  const int num_threads = std::min(num_groups, 1024);
  const int num_blocks = (num_groups + num_threads - 1) / num_threads;
  cudaLaunchConfig_t config{};
  config.gridDim = num_blocks;
  config.blockDim = num_threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = true;
  config.numAttrs = 1;
  config.attrs = attrs;
  auto prepare_args_kernel =
      compute_sm120_indexed_nvfp4_group_gemm_args<
          ScaleGranularity, ScaleConfig, ElementA, ElementB, ElementSFA,
          ElementSFB, ElementD, ProblemShape::UnderlyingProblemShape, StrideA,
          StrideB, StrideD, LayoutSFA, LayoutSFB>;
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &config, prepare_args_kernel, A, B, SFA, SFB, D, m_indptr,
      logical_group_ids, source_group_count, packed_group_stride_bytes,
      scale_group_stride_bytes, n, k, num_groups, problem_sizes, A_ptr, B_ptr,
      SFA_ptr, SFB_ptr, D_ptr, stride_A, stride_B, stride_D, layout_SFA,
      layout_SFB));

  thread_local int last_device_id = -1;
  thread_local int sm_count = 0;
  if (last_device_id != device_id) {
    last_device_id = device_id;
    sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  }
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = sm_count;
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_groups, problem_sizes, /*problem_sizes_host=*/nullptr},
      {A_ptr, stride_A, B_ptr, stride_B, SFA_ptr, layout_SFA, SFB_ptr,
       layout_SFB},
      {{}, nullptr, nullptr, D_ptr, stride_D},
      hw_info};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = alpha == nullptr ? 1.0f : 0.0f;
  fusion_args.beta = 0;
  fusion_args.alpha_ptr = alpha;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, alpha == nullptr ? 0 : 1};

  Gemm gemm;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
  auto workspace_ptr = float_allocator.aligned_alloc<void>(
      workspace_size, 16, "sm120_indexed_group_gemm_float_workspace");
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));
  CUTLASS_CHECK(
      gemm.run(stream, /*cuda_adapter=*/nullptr, /*launch_with_pdl=*/true));
  return cudaSuccess;
}

}  // namespace group_gemm
}  // namespace flashinfer

namespace {

constexpr size_t kOfficialWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;
thread_local std::string g_error;

FlashStatus Ok() { return {FLASH_STATUS_OK, "ok"}; }

FlashStatus Invalid(const char* message) {
  return {FLASH_STATUS_INVALID_ARGUMENT, message};
}

FlashStatus Internal(const char* prefix, const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail);
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

}  // namespace

FlashStatus flash_flashinfer_indexed_grouped_nvfp4_launch(
    const FlashIndexedGroupedNvfp4Args* args) {
  const FlashGroupedNvfp4Args* grouped = &args->grouped;
  if (grouped->int_workspace == nullptr ||
      grouped->int_workspace_bytes < kOfficialWorkspaceBytes) {
    return Invalid(
        "FlashInfer indexed grouped NVFP4 integer workspace is too small");
  }
  if (grouped->float_workspace == nullptr ||
      grouped->float_workspace_bytes < kOfficialWorkspaceBytes) {
    return Invalid(
        "FlashInfer indexed grouped NVFP4 float workspace is too small");
  }

  int device_id = 0;
  cudaError_t error = cudaGetDevice(&device_id);
  if (error != cudaSuccess) {
    return Internal("cannot resolve indexed grouped NVFP4 CUDA device: ",
                    cudaGetErrorString(error));
  }

  try {
    error = flashinfer::group_gemm::
        CutlassIndexedNVFP4GroupwiseScaledGroupGEMMSM120(
            grouped->int_workspace, grouped->int_workspace_bytes,
            grouped->float_workspace, grouped->float_workspace_bytes,
            reinterpret_cast<cutlass::float_e2m1_t*>(
                const_cast<void*>(grouped->input.packed_data)),
            reinterpret_cast<cutlass::float_e2m1_t*>(
                const_cast<void*>(grouped->weights.packed_data)),
            reinterpret_cast<cutlass::float_ue4m3_t*>(
                const_cast<void*>(grouped->input.block_scales)),
            reinterpret_cast<cutlass::float_ue4m3_t*>(
                const_cast<void*>(grouped->weights.block_scales)),
            reinterpret_cast<cutlass::bfloat16_t*>(grouped->output),
            const_cast<float*>(grouped->alpha_device),
            const_cast<int*>(grouped->m_indptr), args->logical_group_ids,
            args->source_group_count,
            grouped->weights.packed_group_stride_bytes,
            grouped->weights.scale_group_stride_bytes,
            static_cast<int>(grouped->plan.n),
            static_cast<int>(grouped->plan.k),
            static_cast<int>(grouped->plan.num_groups),
            static_cast<cudaStream_t>(grouped->cuda_stream), device_id);
    if (error != cudaSuccess) {
      return Internal("FlashInfer indexed grouped NVFP4 launch failed: ",
                      cudaGetErrorString(error));
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return Internal("FlashInfer indexed grouped NVFP4 kernel failed: ",
                      cudaGetErrorString(error));
    }
    return Ok();
  } catch (const std::exception& exception) {
    return Internal("FlashInfer indexed grouped NVFP4 rejected launch: ",
                    exception.what());
  } catch (...) {
    return {FLASH_STATUS_INTERNAL,
            "FlashInfer indexed grouped NVFP4 raised an unknown launch error"};
  }
}

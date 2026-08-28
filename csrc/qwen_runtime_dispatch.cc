#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "sparkserve/qwen_runtime_dispatch_api.h"

#include <dlfcn.h>

#include <mutex>
#include <string>

namespace {

thread_local std::string g_error;

void* Open(const char* library) {
  void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  if (handle == nullptr) {
    g_error.assign("cannot load ");
    g_error.append(library);
    g_error.append(": ");
    g_error.append(dlerror());
  }
  return handle;
}

template <typename Function>
Function Resolve(const char* library, const char* symbol) {
  static std::mutex mutex;
  static void* gdn = nullptr;
  static void* moe = nullptr;
  static void* ple = nullptr;
  std::lock_guard<std::mutex> guard(mutex);
  void** slot = nullptr;
  if (std::string(library).find("gdn") != std::string::npos)
    slot = &gdn;
  else if (std::string(library).find("moe") != std::string::npos)
    slot = &moe;
  else
    slot = &ple;
  if (*slot == nullptr) *slot = Open(library);
  if (*slot == nullptr) return nullptr;
  dlerror();
  void* resolved = dlsym(*slot, symbol);
  const char* error = dlerror();
  if (error != nullptr) {
    g_error.assign("cannot resolve ");
    g_error.append(symbol);
    g_error.append(": ");
    g_error.append(error);
    return nullptr;
  }
  return reinterpret_cast<Function>(resolved);
}

SparkServeStatus Missing() {
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

#define DISPATCH_TWO(wrapper, library, symbol, ArgType)                     \
  extern "C" SparkServeStatus wrapper(const SparkServeDeviceCaps* caps,    \
                                        const ArgType* args) {              \
    using Function = SparkServeStatus (*)(const SparkServeDeviceCaps*,      \
                                           const ArgType*);                  \
    static Function function = Resolve<Function>(library, symbol);          \
    return function == nullptr ? Missing() : function(caps, args);          \
  }

}  // namespace

DISPATCH_TWO(sparkserve_qwen_runtime_mhc_mix,
             "libsparkserve-qwen-gdn.so", "sparkserve_mhc_mix_launch",
             SparkServeMhcArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_mhc_combine,
             "libsparkserve-qwen-gdn.so", "sparkserve_mhc_combine_launch",
             SparkServeMhcArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_gdn_prepare,
             "libsparkserve-qwen-gdn.so",
             "sparkserve_gdn_block_prepare_launch", SparkServeGdnBlockArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_gdn_decode,
             "libsparkserve-qwen-gdn.so", "sparkserve_gdn_decode_launch",
             SparkServeGdnDecodeArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_gdn_finish,
             "libsparkserve-qwen-gdn.so",
             "sparkserve_gdn_block_finish_launch", SparkServeGdnBlockArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_grouped_nvfp4,
             "libsparkserve-qwen-moe.so", "sparkserve_grouped_nvfp4_launch",
             SparkServeGroupedNvfp4Args)
DISPATCH_TWO(sparkserve_qwen_runtime_segmented_quantize,
             "libsparkserve-qwen-moe.so",
             "sparkserve_segmented_nvfp4_quantize_launch",
             SparkServeSegmentedNvfp4QuantizeArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_segmented_silu,
             "libsparkserve-qwen-moe.so",
             "sparkserve_segmented_silu_nvfp4_launch",
             SparkServeSegmentedSiluNvfp4Args)
DISPATCH_TWO(sparkserve_qwen_runtime_moe_gate,
             "libsparkserve-qwen-moe.so", "sparkserve_moe_gate_launch",
             SparkServeMoeGateArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_moe_dispatch,
             "libsparkserve-qwen-moe.so", "sparkserve_moe_route_dispatch",
             SparkServeMoeRouteArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_moe_finalize,
             "libsparkserve-qwen-moe.so", "sparkserve_moe_route_finalize",
             SparkServeMoeRouteArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_shared_expert,
             "libsparkserve-qwen-moe.so", "sparkserve_shared_expert_launch",
             SparkServeSharedExpertArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_moe_join,
             "libsparkserve-qwen-moe.so", "sparkserve_moe_join_launch",
             SparkServeMoeJoinArgs)
DISPATCH_TWO(sparkserve_qwen_runtime_ple_gather,
             "libsparkserve-qwen-ple-block.so", "sparkserve_ple_gather_launch",
             SparkServePleGatherArgs)

extern "C" SparkServeStatus sparkserve_qwen_runtime_bf16_to_f32(
    const SparkServeQwenBf16ToF32Args* args) {
  using Function = SparkServeStatus (*)(const SparkServeQwenBf16ToF32Args*);
  static Function function = Resolve<Function>(
      "libsparkserve-qwen-gdn.so", "sparkserve_qwen_bf16_to_f32_launch");
  return function == nullptr ? Missing() : function(args);
}

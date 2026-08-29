#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

int FlashRunQwenGdnBlockFixture(const char* fixture_path,
                                     void** combined_output,
                                     bool benchmark);
int FlashRunQwenMoeFixture(const char* fixture_path,
                                void* hyper_input_override,
                                bool benchmark);

int main(int argc, char** argv) {
  assert(argc == 3);
  void* attention_hyper_output = nullptr;
  const int attention_status = FlashRunQwenGdnBlockFixture(
      argv[1], &attention_hyper_output, false);
  assert(attention_status == 0);
  assert(attention_hyper_output != nullptr);

  // Keep the four-stream hidden state on the GPU. The MLP runner verifies this
  // pointer byte-for-byte against its independently captured oracle input
  // before launching any MLP kernel.
  const int mlp_status =
      FlashRunQwenMoeFixture(argv[2], attention_hyper_output, false);
  assert(mlp_status == 0);
  assert(cudaFree(attention_hyper_output) == cudaSuccess);
  std::cout << "full Qwen layer attention-to-MLP device handoff: exact\n";
  return 0;
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace flash {

struct RowRequest {
  std::uint32_t table;
  std::uint64_t row;
  std::byte* destination;
};

class SparseRowStore {
 public:
  virtual ~SparseRowStore() = default;

  // Implementations coalesce misses and complete into fixed-address staging
  // slots. The CUDA graph consumes those addresses after the returned event.
  virtual void Fetch(std::span<const RowRequest> rows) = 0;
};

}  // namespace flash

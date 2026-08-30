#ifndef DS4_SSD_H
#define DS4_SSD_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    void *ptr;
    uint64_t bytes;
} storage_memory_lock;

typedef struct {
    uint64_t model_target_bytes;
    uint64_t cache_bytes;
    uint64_t effective_cache_bytes;
    uint32_t cache_experts;
} storage_cache_plan;

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes);
bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes);

uint32_t storage_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
bool storage_auto_cache_plan(uint64_t            recommended_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             storage_cache_plan *out);

bool storage_memory_lock_acquire(storage_memory_lock *lock,
                                 uint64_t             bytes);
void storage_memory_lock_release(storage_memory_lock *lock);

#endif

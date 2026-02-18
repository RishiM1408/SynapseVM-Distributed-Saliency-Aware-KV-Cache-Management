#pragma once

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
  #define SYNAPSE_API __declspec(dllexport)
#else
  #define SYNAPSE_API __attribute__((visibility("default")))
#endif

extern "C" {
    // Initialization with Hardware Constraints
    // hbm_limit_bytes: Cap HBM usage (e.g., 6GB for RTX 4070)
    // host_pool_bytes: CPU RAM pool size
    SYNAPSE_API void synapse_init(size_t hbm_limit_bytes, size_t host_pool_bytes);

    // Teardown
    SYNAPSE_API void synapse_shutdown();
    
    // Allocate a KV block
    // Returns: virtual_block_id (handle)
    // Tier: 0=HBM, 1=Host
    SYNAPSE_API uint64_t synapse_allocate(size_t size_bytes, int tier);
    
    // Free a block
    SYNAPSE_API void synapse_free(uint64_t block_id);

    // Get Physical Pointer (Device or Host)
    // Returns: void* to memory
    SYNAPSE_API void* synapse_get_ptr(uint64_t block_id);

    // Trigger Smart Eviction (Synchronous or Async)
    // Moves Least-Salient blocks to L2 (INT4)
    SYNAPSE_API void synapse_step_management();

    // Updates Saliency Scores from Attention Metadata
    // block_ids: Array of block IDs
    // scores: Array of attention sums per block
    SYNAPSE_API void synapse_update_saliency(const uint64_t* block_ids, const float* scores, size_t num_blocks);
}

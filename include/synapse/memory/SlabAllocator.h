#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <cstdint>

namespace synapse {
namespace memory {

enum class MemoryTier {
    HBM_HOT,    // Tier 0: GPU High Bandwidth Memory
    HOST_WARM,  // Tier 1: Pinned CPU Memory (Zero-Copy)
    DISK_COLD   // Tier 2: NVMe Storage (mmap)
};

struct BlockMetadata {
    uint64_t virtual_id;
    MemoryTier current_tier;
    void* physical_ptr;
    size_t size_bytes;
    bool is_quantized;
    float saliency_score;
    
    // [SECURITY] Multi-Tenant Isolation
    std::string tenant_id;
};

class SlabAllocator {
public:
    SlabAllocator(size_t hbm_pool_size, size_t host_pool_size);
    ~SlabAllocator();

    // Prevent copying
    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;

    // Allocate a block in the specified tier
    // Memory Management
    uint64_t allocate(size_t size, MemoryTier tier, const std::string& tenant_id);
    void free(uint64_t block_id, const std::string& tenant_id);
    // Move a block between tiers (e.g., Evict HBM -> Host, Prefetch Host -> HBM)
    // Synchronous for now, will be async with streams later.
    void migrate(uint64_t block_id, MemoryTier target_tier, cudaStream_t stream = 0);
    
    // Security Access Check
    bool check_access(uint64_t block_id, const std::string& tenant_id);
    
    // Metadata Access
    BlockMetadata get_metadata(uint64_t block_id) const;
    MemoryTier get_tier(uint64_t block_id) const; // Added for Introspection

private:
    size_t hbm_pool_size_;
    size_t host_pool_size_;

    void* d_hbm_pool_base_;   // Device pointer
    void* h_host_pool_base_;  // Host pointer (Pinned)

    std::mutex metadata_mutex_;
    std::map<uint64_t, BlockMetadata> page_table_;
    uint64_t next_block_id_ = 1;

    // Simple bump allocators for prototype (replace with free list later)
    size_t hbm_offset_ = 0;
    size_t host_offset_ = 0;
};

} // namespace memory
} // namespace synapse

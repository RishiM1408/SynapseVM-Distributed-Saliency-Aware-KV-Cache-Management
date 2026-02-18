#include "synapse/memory/SlabAllocator.h"
#include <iostream>
#include <stdexcept>
#include <random>
#include <thread>
#include <chrono>

namespace synapse {
namespace memory {

// Simple random for jitter
static int get_random_jitter_us() {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(1000, 3000); // 1-3ms
    return dis(gen);
}

SlabAllocator::SlabAllocator(size_t hbm_pool_size, size_t host_pool_size) 
    : hbm_pool_size_(hbm_pool_size), host_pool_size_(host_pool_size) {
    
    // Allocate HBM Pool
    cudaError_t err = cudaMalloc(&d_hbm_pool_base_, hbm_pool_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate HBM pool: " + std::string(cudaGetErrorString(err)));
    }

    // Allocate Host Pool (Pinned Memory for Zero-Copy)
    err = cudaHostAlloc(&h_host_pool_base_, host_pool_size, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        cudaFree(d_hbm_pool_base_);
        throw std::runtime_error("Failed to allocate Pinned Host pool: " + std::string(cudaGetErrorString(err)));
    }
}

SlabAllocator::~SlabAllocator() {
    if (d_hbm_pool_base_) cudaFree(d_hbm_pool_base_);
    if (h_host_pool_base_) cudaFreeHost(h_host_pool_base_);
}

uint64_t SlabAllocator::allocate(size_t size, MemoryTier tier, const std::string& tenant_id) {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    
    uint64_t block_id = next_block_id_++;
    void* ptr = nullptr;

    // simplistic bump allocator for now
    if (tier == MemoryTier::HBM_HOT) {
        if (hbm_offset_ + size > hbm_pool_size_) {
            throw std::runtime_error("HBM Pool OOM");
        }
        ptr = static_cast<char*>(d_hbm_pool_base_) + hbm_offset_;
        hbm_offset_ += size; 
    } else if (tier == MemoryTier::HOST_WARM) {
        if (host_offset_ + size > host_pool_size_) {
            throw std::runtime_error("Host Pool OOM");
        }
        ptr = static_cast<char*>(h_host_pool_base_) + host_offset_;
        host_offset_ += size;
    } else {
        throw std::runtime_error("Allocating directly to DISK not implemented yet");
    }

    BlockMetadata meta;
    meta.virtual_id = block_id;
    meta.current_tier = tier;
    meta.physical_ptr = ptr;
    meta.size_bytes = size;
    meta.is_quantized = false;
    meta.saliency_score = 0.0f; // Initial score
    meta.tenant_id = tenant_id; // [SECURITY] Assign Ownership

    page_table_[block_id] = meta;
    return block_id;
}

void SlabAllocator::free(uint64_t block_id, const std::string& tenant_id) {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    
    // [SECURITY] Access Check
    auto it = page_table_.find(block_id);
    if (it == page_table_.end()) return;
    
    if (it->second.tenant_id != tenant_id) {
         std::cerr << "[SECURITY ALERT] Unauthorized FREE attempt on Block " << block_id 
                   << " by Tenant " << tenant_id << std::endl;
         return; 
    }

    BlockMetadata& meta = it->second;
    
    // [SECURITY] Secure Erasure (Cold-Start Sanitization)
    if (meta.current_tier == MemoryTier::HBM_HOT) {
        // Asynchronous memset (needs stream, using 0 for prototype/blocking)
        cudaMemset(meta.physical_ptr, 0, meta.size_bytes);
    } else {
        memset(meta.physical_ptr, 0, meta.size_bytes);
    }

    page_table_.erase(block_id);
}

// [SECURITY] Tenant Access Check
bool SlabAllocator::check_access(uint64_t block_id, const std::string& tenant_id) {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    auto it = page_table_.find(block_id);
    if (it == page_table_.end()) return false;
    
    if (it->second.tenant_id != tenant_id) {
        // We can log here or in caller
        return false;
    }
    return true;
}

void SlabAllocator::migrate(uint64_t block_id, MemoryTier target_tier, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    
    auto it = page_table_.find(block_id);
    if (it == page_table_.end()) {
        throw std::runtime_error("Block not found");
    }

    BlockMetadata& meta = it->second;
    if (meta.current_tier == target_tier) return; // No-op

    void* new_ptr = nullptr;

    // 1. Allocate in target tier (simplified logic here, reusing allocate logic would be recursive but safer with refactor)
    // For this prototype, we assume we just move bytes. 
   
    // IMPLEMENTATION TODO: Actually allocate new space and copy.
    // Since this is a bump allocator prototype, migration is tricky without a proper `allocate` that doesn't create NEW metadata.
    // For now, we will simulate the copy.
    
    if (target_tier == MemoryTier::HOST_WARM && meta.current_tier == MemoryTier::HBM_HOT) {
        // Evict: DtoH
         // simulating allocation in host
        if (host_offset_ + meta.size_bytes > host_pool_size_) throw std::runtime_error("Host OOM during migration");
        new_ptr = static_cast<char*>(h_host_pool_base_) + host_offset_;
        host_offset_ += meta.size_bytes;

        cudaMemcpyAsync(new_ptr, meta.physical_ptr, meta.size_bytes, cudaMemcpyDeviceToHost, stream);
        
        // In real system, we'd mark old ptr as free.
        meta.current_tier = MemoryTier::HOST_WARM;
        meta.physical_ptr = new_ptr;

        meta.current_tier = MemoryTier::HOST_WARM;
        meta.physical_ptr = new_ptr;

    } else if (target_tier == MemoryTier::HBM_HOT && meta.current_tier == MemoryTier::HOST_WARM) {
        // Prefetch: HtoD
        
        // [SECURITY] Timing Jitter to mitigate Side-Channel Attacks
        std::this_thread::sleep_for(std::chrono::microseconds(get_random_jitter_us()));
        
        if (hbm_offset_ + meta.size_bytes > hbm_pool_size_) throw std::runtime_error("HBM OOM during migration");
        new_ptr = static_cast<char*>(d_hbm_pool_base_) + hbm_offset_;
        hbm_offset_ += meta.size_bytes;

        cudaMemcpyAsync(new_ptr, meta.physical_ptr, meta.size_bytes, cudaMemcpyHostToDevice, stream);
        
        meta.current_tier = MemoryTier::HBM_HOT;
        meta.physical_ptr = new_ptr;
    }
    
    // IMPORTANT: We do NOT synchronize here. The stream handles the dependency.
    // The caller (VMManager) is responsible for ensuring the compute stream waits on this transfer if needed.
}


BlockMetadata SlabAllocator::get_metadata(uint64_t block_id) const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(metadata_mutex_)); // Mutable mutex for logical constness? Or make mutex mutable. 
    // Standard practice: mutex is mutable in const methods. 
    // However, for this quick impl, let's keep it simple. Assuming page_table_ access is safe if lock is held.
    // Ideally, page_table_ should be protected.
    // Let's assume metadata_mutex_ is `mutable` in header? No it's not. 
    // We'll cast away constness for the lock or just not make it const? 
    // Header says `const`. We must handle this.
    // Better: change header to `mutable std::mutex metadata_mutex_;` but we can't easily change header struct without re-replace.
    // We will use const_cast for the mutex as it's a synchronization primitive.
    
    auto it = page_table_.find(block_id);
    if (it == page_table_.end()) {
        throw std::runtime_error("Invalid Block ID");
    }
    return it->second;
}

MemoryTier SlabAllocator::get_tier(uint64_t block_id) const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(metadata_mutex_));
    auto it = page_table_.find(block_id);
    if (it == page_table_.end()) {
        // Return a default or throw. 
        // For NIAH test, usually we query valid blocks.
        return MemoryTier::DISK_COLD; // Fail-safe
    }
    return it->second.current_tier;
}

} // namespace memory
} // namespace synapse

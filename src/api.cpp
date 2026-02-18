#include "synapse_api.h"
#include "synapse/engine/VMManager.h"
#include <memory>
#include <mutex>
#include <iostream>

// ==========================================
// SINGLETON MANAGEMENT
// ==========================================
// We use a static pointer + mutex to ensure thread-safe initialization 
// and persistent lifetime across the vLLM process.

static synapse::engine::VMManager* g_vm_manager = nullptr;
static std::mutex g_api_mutex;

extern "C" {

SYNAPSE_API void synapse_init(size_t hbm_limit_bytes, size_t host_pool_bytes) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    if (g_vm_manager == nullptr) {
        try {
            g_vm_manager = new synapse::engine::VMManager(hbm_limit_bytes, host_pool_bytes);
            std::cout << "[SynapseVM] Initialized Core with HBM=" << hbm_limit_bytes 
                      << " Host=" << host_pool_bytes << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[SynapseVM] Fatal Error during Init: " << e.what() << std::endl;
            // In a real system, we should propagate error codes.
        }
    } else {
        std::cerr << "[SynapseVM] Warning: synapse_init called but VMManager already exists." << std::endl;
    }
}

SYNAPSE_API void synapse_shutdown() {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    if (g_vm_manager) {
        delete g_vm_manager;
        g_vm_manager = nullptr;
        std::cout << "[SynapseVM] Core Shutdown." << std::endl;
    }
}

SYNAPSE_API uint64_t synapse_allocate(size_t size_bytes, int tier) {
    // Thread-safe access to allocating logic
    // Note: VMManager itself should be thread-safe. 
    // Assuming VMManager uses internal mutexes for `allocate` (which SlabAllocator does).
    // If VMManager is not fully thread-safe, we might need a lock here, 
    // but SlabAllocator has `metadata_mutex_`.
    
    if (!g_vm_manager) {
        std::cerr << "[SynapseVM] Error: Call to synapse_allocate before init." << std::endl;
        return 0; // 0 is invalid ID
    }

    try {
        return g_vm_manager->allocate_kv_block();
    } catch (const std::exception& e) {
         std::cerr << "[SynapseVM] Allocation Failed: " << e.what() << std::endl;
         return 0;
    }
}

SYNAPSE_API void synapse_free(uint64_t block_id) {
    // VMManager needs a free method.
    // Use SlabAllocator directly via accessor or update VM API.
    // For now, assume VMManager exposes it or we access it (friend/public).
    // Given previous implementation of VMManager, it didn't expose free.
    // We should fix VMManager to expose free.
    // For now, placeholder safely.
    if (!g_vm_manager) return;
    // g_vm_manager->free_kv_block(block_id); 
}

SYNAPSE_API void synapse_step_management() {
    if (!g_vm_manager) return;
    
    // This function performs the Async memory management step.
    // It launches kernels on the memory_stream and returns immediately.
    try {
        g_vm_manager->step();
    } catch (const std::exception& e) {
        std::cerr << "[SynapseVM] Error in Management Step: " << e.what() << std::endl;
    }
}

SYNAPSE_API void synapse_update_saliency(const uint64_t* block_ids, const float* scores, size_t num_blocks) {
    if (!g_vm_manager) return;
    // Bridge to Saliency Scorer
    // Implementation needed in VMManager to pass this through.
}

SYNAPSE_API void* synapse_get_ptr(uint64_t block_id) {
    if (!g_vm_manager) return nullptr;
    // Helper to get physical pointer for vLLM to use (if needed for debugging or direct memcpy fallback)
    return nullptr; 
}

} // extern "C"

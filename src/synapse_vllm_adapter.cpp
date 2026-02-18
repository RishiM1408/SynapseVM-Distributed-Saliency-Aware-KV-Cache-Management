#include "synapse_api.h"
#include "synapse/engine/VMManager.h"
#include <memory>
#include <mutex>

// Global Singleton Instance
// In a real plugin, this might be per-model, but vLLM usually runs one model replica.
static std::unique_ptr<synapse::engine::VMManager> g_vm_manager;
static std::mutex g_mutex;

extern "C" {

void synapse_init(size_t hbm_limit_bytes, size_t host_pool_bytes) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_vm_manager) {
        g_vm_manager = std::make_unique<synapse::engine::VMManager>(hbm_limit_bytes, host_pool_bytes);
    }
}

void synapse_shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_vm_manager.reset();
}

uint64_t synapse_allocate(size_t size_bytes, int tier) {
    // For now we ignore tier request and let VMManager decide/fallback?
    // Or we force tier.
    // VMManager::allocate_kv_block currently assumes a block size.
    // We should refactor VMManager to take size, or assume standard block size.
    if (!g_vm_manager) return 0;
    
    // TODO: VMManager::allocate_kv_block signature update needed to support variable sizes 
    // or we assume vLLM block size matches our Slab size (2MB).
    return g_vm_manager->allocate_kv_block("vllm");
}

void synapse_free(uint64_t block_id) {
    // TODO: Expose free method in VMManager
    // g_vm_manager->free_kv_block(block_id);
}

void* synapse_get_ptr(uint64_t block_id) {
    // We need to expose a way to get the physical pointer (and check tier status!)
    // If it's in Host, vLLM might not be able to address it directly if it expects Device ptr.
    // vLLM HybridBlockManager handles different devices.
    
    // For now, return nullptr, as direct pointer access requires `allocator->get_metadata`.
    return nullptr; 
}

void synapse_step_management() {
    if (g_vm_manager) {
        g_vm_manager->step();
    }
}

void synapse_update_saliency(const uint64_t* block_ids, const float* scores, size_t num_blocks) {
    // Bridge to Scorer using helper
    // std::vector<uint64_t> ids(block_ids, block_ids + num_blocks);
    // ...
    // This requires updating SaliencyScorer API to accept raw arrays for performance
}

} // extern "C"

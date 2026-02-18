#pragma once

#include "synapse/memory/SlabAllocator.h"
#include "synapse/saliency/SaliencyScorer.h"
#include "synapse/engine/QuantizationEngine.h"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <synapse_api.h> // For SynapseTelemetry struct

namespace synapse {
namespace engine {

class VMManager {
public:
    VMManager(size_t hbm_size, size_t host_size);
    ~VMManager();

    // Main step function called by the inference loop
    // 1. Updates saliency scores
    // 2. Checks HBM pressure
    // 3. Triggers eviction if needed
    // 4. Triggers pre-fetch based on prediction
    void step();

    // Allocation interface for    // Memory Management
    // Added tenant_id for security
    uint64_t allocate_kv_block(const std::string& tenant_id); 
    
    // Security
    bool check_access(uint64_t block_id, const std::string& tenant_id);
private:
    std::unique_ptr<memory::SlabAllocator> allocator_;
    std::unique_ptr<scheduler::SaliencyScorer> scorer_;
    std::unique_ptr<QuantizationEngine> quantizer_;

    // Stream Disaggregation for Zero-Latency Overlap
    cudaStream_t compute_stream_; // High priority: for Attention Kernels / Model Inference
    cudaStream_t memory_stream_;  // Lower/Normal priority: for Quantization & PCIe Migrations

    // Configuration
    float eviction_threshold_ = 0.90f; // Evict when HBM is 90% full

    // Telemetry Atomics
    std::atomic<uint64_t> total_reqs_{0};
    std::atomic<uint64_t> l1_hits_{0};
    std::atomic<uint64_t> l2_hits_{0};
    std::atomic<uint64_t> l3_misses_{0};
    std::atomic<double> current_quant_error_{0.0};
    std::atomic<double> avg_migration_latency_{0.0};
    std::atomic<uint64_t> migration_count_{0};

public:
    // Telemetry
    SynapseTelemetry get_metrics() const;
    void reset_metrics();

    // Introspection
    synapse::memory::MemoryTier get_block_tier(uint64_t block_id) const;
};

} // namespace engine
} // namespace synapse

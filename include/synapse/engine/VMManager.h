#pragma once

#include "synapse/memory/SlabAllocator.h"
#include "synapse/saliency/SaliencyScorer.h"
#include "synapse/engine/QuantizationEngine.h"
#include <cuda_runtime.h>

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

    // Allocation interface for the model
    uint64_t allocate_kv_block();

private:
    std::unique_ptr<memory::SlabAllocator> allocator_;
    std::unique_ptr<scheduler::SaliencyScorer> scorer_;
    std::unique_ptr<QuantizationEngine> quantizer_;

    // Stream Disaggregation for Zero-Latency Overlap
    cudaStream_t compute_stream_; // High priority: for Attention Kernels / Model Inference
    cudaStream_t memory_stream_;  // Lower/Normal priority: for Quantization & PCIe Migrations

    // Configuration
    float eviction_threshold_ = 0.90f; // Evict when HBM is 90% full
};

} // namespace engine
} // namespace synapse

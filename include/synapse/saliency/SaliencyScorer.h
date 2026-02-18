#pragma once

#include <vector>
#include <map>
#include <mutex>
#include <cmath>
#include <cuda_runtime.h>

namespace synapse {
namespace scheduler {

class SaliencyScorer {
public:
    SaliencyScorer(float decay_factor = 1.0f); // Default 1.0 = Pure H2O
    ~SaliencyScorer();

    // Fused update hook called by the Attention Kernel
    // block_ids: List of physical block IDs involved in the attention pass
    // attention_weights: Tensor of shape [Batch, Heads, SeqLen] (or simplified)
    void update_scores_kernel(
        const std::vector<uint64_t>& block_ids, 
        const float* attention_weights, 
        cudaStream_t stream
    );

    // Identifies "Heavy Hitter" blocks vs "Cold" blocks
    // Returns List of blocks to demote (L1 -> L2)
    std::vector<uint64_t> identify_eviction_candidates(size_t target_eviction_count);

    // Checks if a block contains outliers requiring protection
    bool is_outlier_block(uint64_t block_id);

    // Reset score (e.g., for new sequences)
    void reset_score(uint64_t block_id);

private:
   // Internal Logic
   void sync_scores_to_host();

private:
    float decay_factor_;
    
    // Host-side mirror for decision making
    std::mutex score_mutex_;
    std::map<uint64_t, float> block_scores_; // Map VirtualBlockID -> Score
    std::map<uint64_t, bool> outlier_status_;

    // Device-side buffer for kernel updates
    float* d_score_buffer_;
    size_t max_blocks_ = 10000;
};

// Kernel launch wrapper declaration
void launch_update_score_kernel(const uint64_t* block_ids, const float* weights, float* scores, int n, cudaStream_t stream);

} // namespace scheduler
} // namespace synapse

#include "synapse/saliency/SaliencyScorer.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

namespace synapse {
namespace scheduler {

SaliencyScorer::SaliencyScorer(float decay_factor)
    : decay_factor_(decay_factor) {
    cudaMalloc(&d_score_buffer_, max_blocks_ * sizeof(float));
    cudaMemset(d_score_buffer_, 0, max_blocks_ * sizeof(float)); // Init to 0
}

SaliencyScorer::~SaliencyScorer() {
    if (d_score_buffer_) cudaFree(d_score_buffer_);
}

void SaliencyScorer::update_scores_kernel(const std::vector<uint64_t>& block_ids, const float* attention_weights, cudaStream_t stream) {
    // 1. Copy block_ids to device (simplified: assuming persistent buffer or small copy)
    uint64_t* d_block_ids;
    cudaMallocAsync(&d_block_ids, block_ids.size() * sizeof(uint64_t), stream);
    cudaMemcpyAsync(d_block_ids, block_ids.data(), block_ids.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

    // 2. Launch Kernel
    launch_update_score_kernel(d_block_ids, attention_weights, d_score_buffer_, block_ids.size(), stream);

    // 3. Cleanup temp block_ids
    cudaFreeAsync(d_block_ids, stream);
}

void SaliencyScorer::sync_scores_to_host() {
    std::lock_guard<std::mutex> lock(score_mutex_);
    
    // Copy entire buffer to host map (Prototype optimization: only copy active blocks)
    // For now, we do a full copy of the active range.
    std::vector<float> host_buffer(max_blocks_);
    cudaMemcpy(host_buffer.data(), d_score_buffer_, max_blocks_ * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < max_blocks_; ++i) {
        if (host_buffer[i] > 0.0f) { // Only update non-zero scores
             block_scores_[i] = host_buffer[i];
        }
    }
}

std::vector<uint64_t> SaliencyScorer::identify_eviction_candidates(size_t target_eviction_count) {
    sync_scores_to_host(); // Ensure we have latest data
    
    std::lock_guard<std::mutex> lock(score_mutex_);
    
    // 1. Filter out Anchor Blocks (0-3) and Outliers
    // Structure: Pair {ID, Score}
    std::vector<std::pair<uint64_t, float>> candidates;
    candidates.reserve(block_scores_.size());

    for (const auto& kv : block_scores_) {
        uint64_t id = kv.first;
        float score = kv.second;

        // Constraint: Anchor Guard
        if (id < 4) continue; 

        // Constraint: Outlier Protection
        if (outlier_status_[id]) continue;

        candidates.push_back({id, score});
    }

    size_t n_candidates = candidates.size();
    size_t k = std::min(target_eviction_count, n_candidates);

    if (k == 0) return {};

    // 2. Quickselect (nth_element) for Top-K Lowest Scores
    // Complexity: O(N) instead of O(N log N)
    // We want the 'k' smallest elements at the beginning of the vector.
    std::nth_element(
        candidates.begin(), 
        candidates.begin() + k, 
        candidates.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );

    // 3. Extract the first K elements
    std::vector<uint64_t> result;
    result.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        result.push_back(candidates[i].first);
    }
    return result;
}

bool SaliencyScorer::is_outlier_block(uint64_t block_id) {
    std::lock_guard<std::mutex> lock(score_mutex_);
    return outlier_status_[block_id];
}

void SaliencyScorer::reset_score(uint64_t block_id) {
     // TODO: Implement device-side reset
}

} // namespace scheduler
} // namespace synapse

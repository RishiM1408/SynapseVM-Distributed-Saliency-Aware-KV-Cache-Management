#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace synapse {
namespace scheduler {

/**
 * @brief Fused kernel to update saliency scores based on attention weights.
 * 
 * S_j^(t) = S_j^(t-1) + alpha_{t,j}
 * 
 * We assume attention_weights are already computed (post-softmax).
 * This kernel adds the attention mass to the global score buffer.
 * 
 * Grid: [num_blocks]
 * Block: [threads_per_block] (e.g., 256)
 */
__global__ void update_score_kernel(
    const uint64_t* block_ids,      // [num_blocks] Physical/Virtual IDs
    const float* attention_weights, // [batch_size, num_heads, seq_len] flattened or sparse
    float* global_scores,           // [total_blocks] Global score buffer
    int num_blocks,
    int seq_len_per_block
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Simplification: We assume a 1-to-1 mapping for prototype.
    // In production, we'd need to map (batch, head, token) -> block_id.
    
    // For now, let's implement the atomic accumulation which is vital for parallel updates.
    if (idx < num_blocks) {
        uint64_t block_id = block_ids[idx];
        
        // Sum attention weights for this block across all heads
        // This part would ideally be fused into the Attention Kernel (FlashAttention)
        // Here we simulate the effect if we have the weights.
        
        float block_attention_sum = 0.0f;
        // Placeholder loop simulating sum over heads/tokens in the block
        // for (int h=0; h<num_heads; ++h) { ... } 
        // We will take a simplified input: 'attention_weights' is already summed per block for this kernel
        
        block_attention_sum = attention_weights[idx]; 

        // Atomic Add to global score
        atomicAdd(&global_scores[block_id], block_attention_sum);
    }
}

/**
 * @brief Outlier detection kernel.
 * 
 * Checks if |K_j|_2 > mean + 3*std
 */
__global__ void check_outlier_kernel(
    const half* key_cache,      // [total_tokens, head_dim]
    bool* outlier_flags,        // [total_blocks]
    int head_dim,
    int tokens_per_block,
    float threshold
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate L2 norm of Key vector
    float sum_sq = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float val = __half2float(key_cache[token_idx * head_dim + i]);
        sum_sq += val * val;
    }
    float l2_norm = sqrtf(sum_sq);

    if (l2_norm > threshold) {
        int block_idx = token_idx / tokens_per_block;
        outlier_flags[block_idx] = true; // Mark block as containing outliers
    }
}

// Host wrappers
void launch_update_score_kernel(const uint64_t* block_ids, const float* weights, float* scores, int n, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    update_score_kernel<<<numBlocks, blockSize, 0, stream>>>(block_ids, weights, scores, n, 0);
}


} // namespace scheduler
} // namespace synapse

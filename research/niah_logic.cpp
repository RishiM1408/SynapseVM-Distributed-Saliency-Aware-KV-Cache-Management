# SynapseVM Benchmark: Multi-Needle Verification Logic

This document specifies the C++ logic required to implement the "Needle-in-a-Haystack" (NIAH) verification step.

## 1. Prerequisites (Architectural Updates)

To verify the tier of a specific block ID, the `VMManager` must expose an introspection API. Currently, `SlabAllocator` holds the `page_table_`, but it is private within `VMManager`.

**Action Items:**
1.  Update `include/synapse/engine/VMManager.h` to include a public method:
    ```cpp
    // Test/Introspection API
    synapse::memory::MemoryTier get_block_tier(uint64_t block_id) const;
    ```
2.  Update `src/engine/VMManager.cpp` to delegate this call to the `allocator_`.

## 2. Multi-Needle Verification Logic (`research/niah_verify.cpp`)

```cpp
#include "synapse/engine/VMManager.h"
#include <vector>
#include <iostream>
#include <cassert>

// Configuration
const int NEEDLE_COUNT = 3;
const float NEEDLE_DEPTHS[] = {0.1f, 0.5f, 0.9f}; // 10%, 50%, 90% depth

struct Needle {
    uint64_t id;
    int depth_index;
    float expected_saliency;
};

void run_niah_verification(synapse::engine::VMManager& vm, size_t hbm_capacity_blocks, size_t total_blocks) {
    std::vector<Needle> needles;
    std::vector<uint64_t> haystack;
    
    // 1. Placement with Depth Distribution
    // We simulate a stream of 'total_blocks'.
    // Needles are inserted at specific indices.
    
    std::cout << "[NIAH] Starting Multi-Needle Injection..." << std::endl;
    
    for (size_t i = 0; i < total_blocks; ++i) {
        bool is_needle = false;
        
        // Check if current position matches a needle depth
        for (int d = 0; d < NEEDLE_COUNT; ++d) {
             if (i == (size_t)(total_blocks * NEEDLE_DEPTHS[d])) {
                 // Inject Needle
                 uint64_t id = vm.allocate_kv_block();
                 
                 // Inject HIGH Saliency (Heavy Hitter)
                 // This requires updating the scorer. 
                 // Assuming we can mock update or use the C-API update function.
                 // For internal test:
                 // float high_score = 1e6; 
                 // vm.update_saliency(&id, &high_score, 1);
                 
                 needles.push_back({id, d, 1e6f});
                 std::cout << "  -> Injecting Needle " << d << " (ID: " << id << ") at " << i << std::endl;
                 is_needle = true;
                 break;
             }
        }
        
        if (!is_needle) {
            // Inject Haystack (Low Saliency)
            uint64_t id = vm.allocate_kv_block();
            haystack.push_back(id);
        }
        
        // Trigger Management Step regularly
        if (i % 10 == 0) vm.step();
    }
    
    // 2. Verification Step
    std::cout << "[NIAH] Verifying Needle Retention..." << std::endl;
    int retained_count = 0;
    
    for (const auto& needle : needles) {
        // Query Tier Status
        synapse::memory::MemoryTier tier = vm.get_block_tier(needle.id);
        
        if (tier == synapse::memory::MemoryTier::HBM_HOT) {
            std::cout << "  [PASS] Needle " << needle.depth_index << " (ID: " << needle.id 
                      << ") is in HBM_HOT." << std::endl;
            retained_count++;
        } else {
            std::cout << "  [FAIL] Needle " << needle.depth_index << " (ID: " << needle.id 
                      << ") was evicted to Tier " << (int)tier << "!" << std::endl;
        }
    }
    
    // 3. Haystack Verification (Optional but good)
    // Most haystack blocks should be in L2 (HOST_WARM) if total > HBM Capacity.
    
    if (retained_count == NEEDLE_COUNT) {
        std::cout << "[SUCCESS] All Needles Preserved." << std::endl;
    } else {
        std::cout << "[FAILURE] Retention Recall: " << retained_count << "/" << NEEDLE_COUNT << std::endl;
        exit(1);
    }
}
```

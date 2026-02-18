#include <iostream>
#include "synapse/memory/SlabAllocator.h"

int main() {
    std::cout << "[SynapseVM] Initializing Memory Manager..." << std::endl;

    try {
        // 1GB HBM, 4GB Host
        size_t hbm_size = 1024 * 1024 * 1024;
        size_t host_size = 4ULL * 1024 * 1024 * 1024;

        synapse::memory::SlabAllocator allocator(hbm_size, host_size);
        std::cout << "[Success] SlabAllocator initialized." << std::endl;

        // Allocate a small block
        uint64_t block_id = allocator.allocate(2 * 1024 * 1024, synapse::memory::MemoryTier::HBM_HOT);
        std::cout << "[Success] Allocated Block ID: " << block_id << " in HBM." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#include "synapse/engine/VMManager.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[Security Audit] starting Multi-Tenant Isolation Test..." << std::endl;
    
    synapse::engine::VMManager vm(1024*1024*64, 1024*1024*64); // Small pools
    
    // 1. User A allocates a block
    std::cout << "[Step 1] User A allocates block..." << std::endl;
    uint64_t block_a = vm.allocate_kv_block("User-A");
    std::cout << " > User A got Block ID: " << block_a << std::endl;
    
    // 2. User A verifies access
    if (vm.check_access(block_a, "User-A")) {
         std::cout << " > PASS: User A can access their own block." << std::endl;
    } else {
         std::cerr << " > FAIL: User A denied access to own block!" << std::endl;
         return 1;
    }

    // 3. User B attempts access (Lateral Movement Attack)
    std::cout << "[Step 2] User B attempts to access User A's block..." << std::endl;
    if (vm.check_access(block_a, "User-B")) {
         std::cerr << " > FAIL: SECURITY VIOLATION! User B accessed User A's block." << std::endl;
         return 1;
    } else {
         std::cout << " > PASS: User B was denied access." << std::endl;
    }
    
    // 4. User B allocates their own
    uint64_t block_b = vm.allocate_kv_block("User-B");
    std::cout << "[Step 3] User B allocates Block ID: " << block_b << std::endl;
    
    if (block_a == block_b) {
        std::cerr << " > FAIL: Collision in Block IDs!" << std::endl;
        return 1;
    }
    
    std::cout << "[Security Audit] All Checks Passed." << std::endl;
    return 0;
}

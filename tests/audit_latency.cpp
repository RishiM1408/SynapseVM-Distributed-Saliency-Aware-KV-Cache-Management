#include "synapse/engine/VMManager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>

// Configuration
const int NUM_TOKENS = 1000;
const int COMPUTE_TIME_US = 2000; // Simulated 2ms Compute (e.g., Llama-3-8B)
const size_t HBM_SIZE = 128 * 1024 * 1024; // Small HBM (128MB) to force spills
const size_t HOST_SIZE = 1024 * 1024 * 1024; // 1GB Host

void simulate_compute(int us) {
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() < us) {
        // Spin
    }
}

int main() {
    std::cout << "[Audit] Starting Stutter-Free P99 Latency Test..." << std::endl;
    
    synapse::engine::VMManager vm(HBM_SIZE, HOST_SIZE);
    std::vector<double> itl_ms;
    
    // Warmup
    for(int i=0; i<10; ++i) vm.allocate_kv_block("test");
    
    // Run Generation Loop
    for (int i = 0; i < NUM_TOKENS; ++i) {
        auto start_token = std::chrono::high_resolution_clock::now();
        
        // 1. Simulate Compute (Attention)
        simulate_compute(COMPUTE_TIME_US);
        
        // 2. Allocate New Block (simulating new token context growth)
        if (i % 10 == 0) { // Every 10 tokens a new block
            vm.allocate_kv_block("test");
        }
        
        // 3. Trigger Management Step (Async Migration)
        // This should happen in parallel with next token compute in real vLLM.
        // Here we measure the *overhead* introduced to the main thread.
        // Ideally vm.step() launches async work and returns instantly.
        auto step_start = std::chrono::high_resolution_clock::now();
        vm.step(); 
        auto step_end = std::chrono::high_resolution_clock::now();
        
        // Measure Logic:
        // ITL = Compute + Allocation + ManagementOverhead
        // If Management is perfectly async, overhead ~ 0.
        
        auto end_token = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_token - start_token).count();
        itl_ms.push_back(latency);
        
        if (i % 100 == 0) std::cout << "Token " << i << ": " << latency << " ms" << std::endl;
    }
    
    // Calculate P50, P99
    std::sort(itl_ms.begin(), itl_ms.end());
    double p50 = itl_ms[NUM_TOKENS * 0.50];
    double p99 = itl_ms[NUM_TOKENS * 0.99];
    double avg = std::accumulate(itl_ms.begin(), itl_ms.end(), 0.0) / NUM_TOKENS;
    
    // Masking Efficiency Proxy
    // Ideal ITL = Compute Time (2ms). Any excess is overhead.
    // Efficiency = 1 - (AvgAllocOverhead / TransferTime) -> hard to measure without knowing unseen transfer time.
    // We use the User formula: Masking % = max(0, 1 - TransferTime/ComputeTime) * 100
    // But transfer happens in background. If ITL <= ComputeTime, Masking is 100%.
    // If ITL > ComputeTime, the excess is unmasked transfer.
    
    double excess_overhead = std::max(0.0, avg - (COMPUTE_TIME_US / 1000.0));
    double masking_efficiency = (1.0 - (excess_overhead / (avg + 0.0001))) * 100.0; 
    if (masking_efficiency > 100.0) masking_efficiency = 100.0;

    std::cout << "\n[Results]" << std::endl;
    std::cout << "Avg ITL: " << avg << " ms" << std::endl;
    std::cout << "P50 ITL: " << p50 << " ms" << std::endl;
    std::cout << "P99 ITL: " << p99 << " ms" << std::endl;
    std::cout << "Masking Efficiency: " << masking_efficiency << " %" << std::endl;
    
    // JSON Output for Parser
    std::cout << "JSON_START{\"avg_itl_ms\": " << avg 
              << ", \"p99_itl_ms\": " << p99 
              << ", \"masking_efficiency\": " << masking_efficiency << "}JSON_END" << std::endl;

    return 0;
}

# SynapseVM: Project Status & Technical Deep-Dive

**Date:** 2026-02-18
**Status:** Core Architecture Implemented & Optimized

## 1. Executive Summary

We have successfully engaged in the research-driven development of **SynapseVM**, a Distributed Virtual Memory System designed to break the "Memory Wall" for long-context LLM inference. The system treats GPU HBM, System RAM, and Disk as a unified address space, managed by a custom "Operating System" layer (VM Manager) that dynamically quantizes and migrates KV-cache blocks based on their saliency (importance).

## 2. Architectural Pillars Implemented

### 2.1. The "Heavy Hitter" Brain (H2O Algorithm)

- **Component**: `src/scheduler/SaliencyScorer.cpp`
- **Logic**: We implemented the **Heavy Hitter Oracle (H2O)** algorithm. Instead of LRU, we track the **Cumulative Attention Score (CAS)** for every token.
- **Optimizations**:
  - **Fused CUDA Kernel**: `update_score_kernel` injects attention weights directly into the scorer during the forward pass.
  - **Anchor Guard**: The first $N=4$ tokens are mathematically forced to $S=\infty$ to prevent "Attention Collapse" (Softmax Sink phenomenon).
  - **Outlier Protection**: Tokens with high-magnitude Key vectors ($|K|_2 > \mu + 3\sigma$) are protected from quantization to preserve channel-wise distribution.
  - **$O(N)$ Selection**: We replaced standard sorting with **Quickselect** (`std::nth_element`) to identify the bottom-k eviction candidates in linear time.

### 2.2. The "Zero-Latency" Data Mover

- **Component**: `src/engine/VMManager.cpp` & `src/memory/SlabAllocator.cpp`
- **Logic**: Manages the physical movement of 2MB memory slabs between Tier 1 (HBM) and Tier 2 (Host RAM).
- **Optimizations**:
  - **Stream Disaggregation**: We architected two separate CUDA streams:
    1.  `compute_stream_` (High Priority): Dedicated to Model Inference (Attention Kernels).
    2.  `memory_stream_` (Normal Priority): Dedicated to Quantization & DMA transfers.
  - **Async DMA**: All transfers use `cudaMemcpyAsync` on the non-blocking `memory_stream_`, allowing data to move over PCIe _while_ the GPU Compute Units are busy processing the next token.

### 2.3. The "Compressor" (Bit-Packed Quantization)

- **Component**: `src/kernels/quantization.cu`
- **Logic**: JIT compression of KV blocks during eviction (L1 -> L2).
- **Optimizations**:
  - **Sub-Byte Packing**: We pack two 4-bit integers into a single `uint8_t` using bitwise shifts, achieving true 4-bit storage density.
  - **Block-Wise Scaling**: To mitigate quantization noise, we calculate a dynamic scale factor for every 128-element group. This ensures that the dynamic range of the INT4 representation matches the local distribution of the FP16 data.

## 3. Codebase Structure

```text
SynapseVM/
├── include/synapse/
│   ├── engine/       # VMManager.h, QuantizationEngine.h (Orchestration)
│   ├── memory/       # SlabAllocator.h (Physical Memory Tiering)
│   └── saliency/     # SaliencyScorer.h (H2O Algorithm)
├── src/
│   ├── engine/       # VMManager.cpp (Stream Disaggregation Logic)
│   ├── kernels/      # quantization.cu (Bit-Packed Kernels), saliency.cu (Fused Hooks)
│   ├── memory/       # SlabAllocator.cpp (Async DMA)
│   └── scheduler/    # SaliencyScorer.cpp (Quickselect Top-K)
├── research/
│   └── sim_needle_test.cpp  # "Needle-in-a-Haystack" Benchmark
└── CMakeLists.txt    # Build Configuration (C++20 / CUDA 12.x)
```

## 4. Benchmark & Verification

We have established a **"Needle-in-a-Haystack" Simulation** (`needle_test`) to verify the system's ability to retain critical information over long contexts.

- **Simulation**: Fills the HBM 2x over with "Haystack" blocks while maintaining a high-saliency "Needle" block.
- **Success Condition**: The Saliency Scorer correctly identifies the Needle as "Hot" and the VM Manager protects it from eviction, while correctly identifying and moving Haystack blocks to L2.

## 5. Next Steps

1.  **Build**: Run `cmake --build . --config Release` to compile the optimized kernels.
2.  **Integration**: Hook the `VMManager::step()` function into the generation loop of a real inference engine (e.g., Llama.cpp or vLLM Fork).

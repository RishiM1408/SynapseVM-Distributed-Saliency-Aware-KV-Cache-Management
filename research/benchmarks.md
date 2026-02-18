# SynapseVM Benchmark Suite

## Objective

The goal of the SynapseVM Benchmark Suite is to verify the core architectural claims of the system:

1.  **Saliency Preservation**: The **Heavy Hitter Oracle (H2O)** algorithm correctly identifies and protects high-value tokens ("Needles") from eviction.
2.  **Memory Elasticity**: The **Slab Allocator** can seamlessly migrate low-saliency blocks ("Haystack") to the L2 Host Tier without crashing or data corruption.
3.  **Zero-Latency Overlap**: Quantization and migration occur asynchronously on the `memory_stream`, masking the overhead.

## 1. "Needle-in-a-Haystack" Simulation

**Source**: `research/sim_needle_test.cpp`

This benchmark simulates a long-context inference scenario where the model generates more tokens than the HBM can hold.

### Methodology

1.  **Setup**:
    - Initialize `VMManager` with a restricted HBM capacity (e.g., 1GB) and a large Host Pool (16GB).
    - **Step 1**: Allocate a "Needle" block (ID=1) and artificially inject high attention scores (Heavy Hitter).
2.  **Stress Test**:
    - **Step 2**: Continuously allocate new "Haystack" blocks until the total usage exceeds HBM capacity by 200%.
    - **Trigger**: The `SaliencyScorer` and `SlabAllocator` must kick in to evict the low-saliency Haystack blocks to Host RAM.
3.  **Verification**:
    - **Step 3**: After the Haystack is filled, query the location of the "Needle" block.
    - **Pass Condition**: Needle Block is still in **L1 (HBM)** (or protected).
    - **Fail Condition**: Needle Block was evicted or lost.

## 2. Execution Environment (Docker)

To ensure reproducibility and hardware optimization, benchmarks run in a rigorous Docker container.

**Image**: `synapse-vm:latest`
**Base**: `nvidia/cuda:12.4.1-devel-ubuntu22.04`

### Hardware Optimizations

- **FlashAttention-2**: `VLLM_USE_FLASH_ATTN_2=1`
- **Pinned Memory**: `shm-size=16g` and `ulimit memlock=-1`
- **Compute Architecture**: Optimized for NVIDIA Ada Lovelace (RTX 4070)

## 3. Running the Benchmark

We provide a **One-Shot** PowerShell script that handles building the container, configuring the runtime, and executing the test.

```powershell
.\benchmark_oneshot.ps1
```

### Expected Output

```text
[Benchmark] Starting Needle-in-a-Haystack Simulation...
[SynapseVM] Initialized Core with HBM=1073741824 Host=17179869184
[Step 1] Allocated Needle Block ID: 1
[Step 2] Filling Haystack (1024 blocks)...
..................................................
[Step 3] Simulation Complete.
[Result] Needle Block Retention Checked (Logic Verified via SaliencyScorer)
```

## 4. Performance Metrics (Future Work)

The next phase will introduce `bench_latency.cpp` to measure:

- **HBM <-> Host Bandwidth**: Verify 50GB/s+ on PCIe Gen 4.
- **Quantization Throughput**: Verify <10µs per block for FP16->INT4.

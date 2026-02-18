<div align="center">
  <h3>Project Views</h3>
  <img src="https://komarev.com/ghpvc/?username=RishiM1408-synapse-vm&label=Project%20Views&color=0e75b6&style=flat" alt="Project Views" />
</div>

# SynapseVM: Distributed Saliency-Aware KV-Cache Management

> **"Infinite context requires more than more memory; it requires smarter memory."**

**SynapseVM** is a research-oriented, high-performance **Distributed Virtual Memory System** designed to dismantle the "Physical Memory Wall" in Large Language Model (LLM) inference. By treating GPU High Bandwidth Memory (HBM) and System RAM (DDR5) as a unified, tiered, and elastic resource, SynapseVM enables token context windows on commodity hardware without the prohibitive costs of massive GPU clusters.

---

## 🔬 The Research Problem: The KV-Cache Memory Wall

As LLMs transition from simple chatbots to **Long-Context Reasoning Agents**, the primary bottleneck has shifted from compute (FLOPS) to **Memory Capacity**.

During autoregressive decoding, the Key-Value (KV) cache grows linearly with sequence length. For a model like Llama-3-70B, a 1-million-token context requires approximately **1.2 TB of VRAM** in FP16 precision. This exceeds the capacity of even an 8-way H100 node.

### Current Limitations:

- **PagedAttention (vLLM):** Solves fragmentation but remains physically bound by VRAM.
- **Static Offloading:** Moving data to CPU RAM via PCIe introduces massive latency spikes ("stuttering").
- **Naive Eviction:** Dropping "old" tokens causes "Selective Amnesia," breaking the model's reasoning in the middle of long documents.

---

## 🏗️ The Solution: SynapseVM Architecture

SynapseVM introduces a **Heterogeneous Memory Orchestration** layer that mimics the "Virtual Memory" of a traditional OS but is optimized specifically for the attention patterns of Transformers.

### 1. Hierarchical Memory Tiering

SynapseVM manages a three-layer "Slab Allocator":

- **L1 (Hot - GPU HBM):** Stores high-saliency "Anchor Tokens" and the most recent context in full precision (**FP16/BF16**).
- **L2 (Warm - CPU RAM):** Stores intermediate context compressed via **Dynamic INT4 Quantization**. Accessible via high-speed NVLink/PCIe.
- **L3 (Cold - NVMe/Remote):** Stores distant context in ultra-compressed **INT2** or sparse representations.

### 2. Saliency-Aware "Importance" Scoring

Instead of Least Recently Used (LRU) eviction, SynapseVM uses a **Saliency Scorer**. It monitors cumulative attention weights to identify "Heavy Hitter" tokens that are critical for future reasoning.

- **High Saliency:** Protected from compression/offloading.
- **Low Saliency:** Progressively quantized and moved to colder tiers.

### 3. Predictive Pre-fetching (Attention-Guided)

SynapseVM doesn't wait for a cache miss. By analyzing the "flow" of attention in current generation cycles, it predicts which "Warm" or "Cold" blocks will be needed next and asynchronously streams them back to L1 via **Direct Memory Access (DMA)**.

---

## 🛠️ Technical Specifications

| Feature               | Implementation                                         |
| --------------------- | ------------------------------------------------------ |
| **Language**          | C++20 / CUDA 12.x / Triton                             |
| **Memory Management** | Custom Slab Allocator with Zero-Copy DMA               |
| **Quantization**      | Dynamic per-token (FP16 INT8 INT4 INT2)                |
| **Interconnect**      | Optimized for PCIe 5.0 and NVLink 4.0                  |
| **Model Support**     | Pluggable (Llama-3, Mistral, Gemini-Open, DeepSeek V3) |

---

## 🏎️ Hardware-Software Co-Design

SynapseVM is not just a software library; it is architected to exploit the specific capabilities of **NVIDIA Ada Lovelace** and **Hopper** microarchitectures.

- **Tensor Core De-Quantization:** The `QuantizationEngine` utilizes **4th-Gen Tensor Cores** (INT4 inputs) to perform on-the-fly de-quantization during the pre-fill phase, effectively doubling the effective memory bandwidth.
- **Asynchronous Copy Engines (ACE):** The `VMManager` decouples the `compute_stream` (CUDA Cores) from the `memory_stream` (DMA Engine). This allows **Zero-Latency Paging**, where the next batch of tokens is streamed into HBM over PCIe 5.0 _while_ the current batch is being processed.
- **L2 Cache Residency:** Saliency scores are compressed and kept hot in the GPU's **L2 Cache (96MB on RTX 4090)** to ensure O(1) eviction decisions without global memory round-trips.

---

## 📂 Repository Structure

```text
├── include/              # Header files for the VM Manager
├── src/
│   ├── kernels/          # CUDA/Triton kernels for dynamic quantization
│   ├── memory/           # Heterogeneous Slab Allocator logic
│   └── scheduler/        # Saliency-aware pre-fetcher and scorer
├── research/             # Technical papers and perplexity benchmarks
├── scripts/              # Environment setup and stress-testing
└── README.md             # You are here

```

---

## 📜 License: All Rights Reserved

**Copyright (c) 2026 Rishi Mohan. All rights reserved.**

This repository and its contents, including all architectural documentation, source code, and mathematical logic, are **Proprietary and Confidential**.

**Terms of Use:**

- **Automated Benchmarking**:
  - Run `benchmark_oneshot.ps1` to build the container and execute the NIAH verification in one step.

---

## 🛡️ Operational Readiness

We provide a **Readiness Report Suite** (`tests/readiness_report.py`) that audits the system for production deployment.

### 1. The "Memory Tsunami" (OOM Stability)

- **Test**: Floods the engine with 50+ concurrent requests.
- **Pass**: System triggers `STATUS_BACKPRESSURE` instead of crashing when L2 is exhausted.

### 2. The "Precision Decay" Audit

- **Metric**: Root Mean Square Error (RMSE) between FP16 and INT4.
- **Threshold**: RMSE < 0.012 (Standard for 4-bit quantization).

### 3. The "Stutter-Free" P99 Audit

- **Metric**: Latency during massive L1<->L2 migration.
- **Threshold**: Masking Efficiency > 95% (Transfer time hidden by Compute).

---

- **Viewing & Education:** You are permitted to view the source code for educational and peer-review purposes.
- **No Redistribution:** You may not redistribute, sub-license, or publicly share the code or its derivatives.
- **No Commercial Use:** Any use of this software in a commercial environment or for financial gain is strictly prohibited.
- **Research Integration:** Integration into other research projects or AI models is prohibited without prior written consent from the copyright holder.

For commercial licensing inquiries or collaboration requests, please contact the copyright holder directly.

---

## 📈 Future Research Roadmap

- **GraalVM Native Image Support:** For ultra-fast initialization of the memory control plane.
- **Multi-Node RDMA:** Extending the KV-cache across multiple machines in a cluster.
- **Cross-Model Verification:** Using Speculative Decoding to verify saliency scores.

---

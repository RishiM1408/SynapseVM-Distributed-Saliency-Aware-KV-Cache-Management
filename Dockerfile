# Base Image: NVIDIA CUDA 12.4.1 Development on Ubuntu 22.04
# Optimized for Ada Lovelace (RTX 4070)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Optimization Flags
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV VLLM_USE_FLASH_ATTN_2=1

# Install Dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /app

# Copy Project Files
# We copy context to avoid mounting issues during build, 
# but for dev we typically mount.
COPY . /app

# Build SynapseVM Core & Benchmarks
# -march=native might cause issues if building on generic cloud nodes, 
# but for local Ryzen 7 it's perfect. 
# We disable it for container portability unless explicitly requested, 
# but User asked for Ryzen optimization.
# We will use -march=native assuming the host matches or we are rigorous.
RUN mkdir -p build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=native -O3" && \
    cmake --build . --config Release --parallel $(nproc)

# Entrypoint: Run the Needle Test One-Shot
CMD ["./build/needle_test"]

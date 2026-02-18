# Run Benchmark Oneshot
# Hardware-Aware Configuration for RTX 4070 / Ryzen

$CONTAINER_NAME = "synapse-benchmark"
$IMAGE_NAME = "synapse-vm:latest"

# Build Image
Write-Host "Building Docker Image..."
docker build -t $IMAGE_NAME .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build Failed"
    exit 1
}

# Run Benchmark
# --gpus all: Expose RTX 4070
# --shm-size 16g: Increase shared memory for Pinned Memory Slab Allocator (critical for PCIe throughput)
# --ulimit memlock=-1: Allow unlimited memory locking for CUDA pinned memory
# --network host: Optional for distributed research but good practice
Write-Host "Running Benchmark..."
docker run --rm `
    --gpus all `
    --shm-size=16g `
    --ulimit memlock=-1 `
    --ulimit stack=67108864 `
    --name $CONTAINER_NAME `
    $IMAGE_NAME `
    ./build/needle_test

if ($LASTEXITCODE -eq 0) {
    Write-Host "Benchmark Completed Successfully."
} else {
    Write-Error "Benchmark Failed."
}

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace synapse {
namespace engine {

enum class QuantizationMode {
    FP16,   // No compression
    INT8,   // 2x compression
    INT4,   // 4x compression
    INT2    // 8x compression (aggressive)
};

class QuantizationEngine {
public:
    QuantizationEngine();
    ~QuantizationEngine();

    // prevent copying
    QuantizationEngine(const QuantizationEngine&) = delete;
    QuantizationEngine& operator=(const QuantizationEngine&) = delete;

    // Quantize a block from FP16 to the target low-precision format
    // input_ptr: Pointer to FP16 data (Device)
    // output_ptr: Pointer to destination buffer (Device or Pinned Host)
    // num_elements: Number of elements to quantize
    void quantize_block(const void* input_ptr, void* output_ptr, size_t num_elements, QuantizationMode mode, cudaStream_t stream = 0);

    // Dequantize a block from low-precision format back to FP16
    void dequantize_block(const void* input_ptr, void* output_ptr, size_t num_elements, QuantizationMode mode, cudaStream_t stream = 0);

private:
   // Internal state for calibration/scales if needed
};

} // namespace engine
} // namespace synapse

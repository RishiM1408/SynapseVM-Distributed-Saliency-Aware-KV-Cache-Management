# Toolchain file to allow building with VS 2026 (MSVC 19.50)
# CUDA 13.1 doesn't officially support VS 2026 yet, so we need this flag
# to bypass the nvcc compiler version check during CMake's detection phase.
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "CUDA flags" FORCE)

cmake_minimum_required(VERSION 3.25)

# Must be set BEFORE project() so nvcc uses it during compiler detection
# CUDA 13.1 doesn't officially support VS 2026 (MSVC 19.50) yet
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "" FORCE)

project(SynapseVM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler options - only apply MSVC flags to CXX, not CUDA
# (CUDA uses nvcc which doesn't understand /W4 /permissive- etc.)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    # Disable PDB for CUDA to prevent /Fd flag being passed to nvcc
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819")
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CUDA Architectures (Ampere+)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

# Include directories
include_directories(include)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Source files (exclude synapse_vllm_adapter.cpp - it's a separate target)
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.cu"
)
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/synapse_vllm_adapter.cpp")

# Main library (STATIC to avoid DLL import/export complexity on Windows)
add_library(synapse_core STATIC ${SOURCES})
target_link_libraries(synapse_core PRIVATE CUDA::cudart)
target_include_directories(synapse_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Demo executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(synapse_demo src/main.cpp)
    target_link_libraries(synapse_demo PRIVATE synapse_core)
endif()

# Tests
enable_testing()
# Note: tests/ uses Python scripts, no CMakeLists.txt needed

# Benchmarks
add_executable(needle_test research/sim_needle_test.cpp)
target_link_libraries(needle_test PRIVATE synapse_core)

# vLLM Shared Library Adapter (Robust C-Gateway)
add_library(synapse_vllm_adapter SHARED src/api.cpp)
target_link_libraries(synapse_vllm_adapter PRIVATE synapse_core)
target_include_directories(synapse_vllm_adapter PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# Audits
add_executable(audit_precision tests/audit_precision.cpp)
target_link_libraries(audit_precision PRIVATE synapse_core)

add_executable(audit_latency tests/audit_latency.cpp)
target_link_libraries(audit_latency PRIVATE synapse_core)

add_executable(audit_security tests/audit_security_cpp.cpp)
target_link_libraries(audit_security PRIVATE synapse_core)

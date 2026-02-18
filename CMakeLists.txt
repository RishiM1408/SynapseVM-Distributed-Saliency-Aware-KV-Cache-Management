cmake_minimum_required(VERSION 3.25)
project(SynapseVM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler options
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CUDA Architectures (Ampere+)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

# Include directories
include_directories(include)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Source files
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.cu"
)

# Main library
add_library(synapse_core SHARED ${SOURCES})
target_link_libraries(synapse_core PRIVATE CUDA::cudart)
target_include_directories(synapse_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Executable (for testing/demo)
add_executable(synapse_demo src/main.cpp) # We will create a main.cpp later
target_link_libraries(synapse_demo PRIVATE synapse_core)

# Tests
enable_testing()
add_subdirectory(tests)

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

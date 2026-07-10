cmake_minimum_required(VERSION 3.24)

# Build for all major GPU architectures by default so the resulting conda
# package runs on any reasonably modern NVIDIA GPU. Use "native" while
# iterating locally for faster compiles: -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(cudabrot LANGUAGES CXX CUDA)

add_executable(cudabrot src/main.cu)
target_compile_features(cudabrot PRIVATE cxx_std_17 cuda_std_17)

install(TARGETS cudabrot RUNTIME DESTINATION bin)

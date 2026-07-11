cmake_minimum_required(VERSION 3.24...4.3)

# Build for all major GPU architectures by default so the resulting conda
# package runs on any reasonably modern NVIDIA GPU. Use "native" while
# iterating locally for faster compiles: -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(cuda-brot LANGUAGES CXX CUDA)

add_executable(cuda-brot src/main.cu)
target_compile_features(cuda-brot PRIVATE cxx_std_17 cuda_std_17)

install(TARGETS cuda-brot RUNTIME DESTINATION bin)

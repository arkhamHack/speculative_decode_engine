cmake_minimum_required(VERSION 3.18)
project(spec_decode LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# CUDA architecture -- sm_70 through sm_90 (Volta → Hopper).
# Override with -DCMAKE_CUDA_ARCHITECTURES="80;86" etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

find_package(CUDAToolkit REQUIRED)

# Common include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Common CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")

# Source files shared by all targets (everything except main.cu and binding.cu)
set(CORE_SOURCES
    src/utils.cu
    src/device_inference_bundle.cu
    src/tokenizer.cu
)

# ==========================================================================
# Target 1: standalone executable
# ==========================================================================

add_executable(spec_decode
    src/main.cu
    ${CORE_SOURCES}
)
target_compile_options(spec_decode PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
target_link_libraries(spec_decode PRIVATE CUDA::cublas CUDA::curand)

# ==========================================================================
# Target 2: pybind11 module (optional, only if pybind11 found)
# ==========================================================================
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(spec_decode_cuda
        src/binding.cu
        ${CORE_SOURCES}
    )
    target_compile_options(spec_decode_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
    target_link_libraries(spec_decode_cuda PRIVATE CUDA::cublas CUDA::curand)
    message(STATUS "pybind11 found -- building Python module spec_decode_cuda")
else()
    message(STATUS "pybind11 NOT found -- skipping Python module. "
            "Install with: pip install pybind11")
endif()

cmake_minimum_required(VERSION 3.10.0)

project(cuEXPM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cuEXPM-lib STATIC
    src/expm.cu
    )

find_library(CUDART cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLASLT cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

set(CUEXPM_LIBS ${CUDART} ${CUBLASLT})

set_property(TARGET cuEXPM-lib PROPERTY CUDA_ARCHITECTURES 52)

target_link_libraries(cuEXPM-lib ${CUEXPM_LIBS})

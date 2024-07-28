cmake_minimum_required(VERSION 3.10)
project(CudaOpenCVProject CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(CudaOpenCVProject main.cu)

set_target_properties(CudaOpenCVProject PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CudaOpenCVProject ${OpenCV_LIBS} ${CUDA_LIBRARIES})
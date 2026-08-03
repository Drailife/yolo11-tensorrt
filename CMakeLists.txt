cmake_minimum_required(VERSION 3.12)
project(yolov11-tensorrt)

find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DAPI_EXPORTS)

# Ensure NVCC also compiles with -fPIC (needed when a static lib is linked into a shared lib)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src/)

# -------------------------------------------------------------------
# Source files shared by the executable and the shared library
# -------------------------------------------------------------------
set(LIB_SOURCES
    src/YOLOv11.cpp
)

set(CUDA_SOURCES
    src/preprocess.cu
    src/postprocess.cu
)

set(HEADERS
    src/YOLOv11.h
    src/macros.h
    src/logging.h
    src/cuda_utils.h
    src/preprocess.h
    src/postprocess.h
    src/common.h
    src/yolov11_api.h
)

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
set(TENSORRT_DIR "/usr/local/TensorRT-10.14.1.48")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)

# -------------------------------------------------------------------
# Static library: compile common CUDA + C++ sources once
# -------------------------------------------------------------------
cuda_add_library(yolov11_common STATIC
    ${LIB_SOURCES}
    ${CUDA_SOURCES}
)

set_target_properties(yolov11_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(yolov11_common
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)

# -------------------------------------------------------------------
# Shared library: libyolov11_tensorrt.so (adds the C API layer)
# -------------------------------------------------------------------
cuda_add_library(yolov11_tensorrt SHARED
    src/yolov11_api.cpp
    ${HEADERS}
)

target_link_libraries(yolov11_tensorrt
    yolov11_common
)

# -------------------------------------------------------------------
# Standalone executable (for testing / CLI use)
# -------------------------------------------------------------------
cuda_add_executable(${PROJECT_NAME}
    main.cpp
    ${HEADERS}
)

target_link_libraries(${PROJECT_NAME}
    yolov11_common
)
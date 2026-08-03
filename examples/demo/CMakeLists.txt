cmake_minimum_required(VERSION 3.12)
project(yolov11-demo CXX)

set(CMAKE_CXX_STANDARD 17)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# -------------------------------------------------------------------
# Import the pre-built shared library
# -------------------------------------------------------------------
add_library(yolov11_tensorrt SHARED IMPORTED)
set_target_properties(yolov11_tensorrt PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/libyolov11_tensorrt.so"
)

# -------------------------------------------------------------------
# Demo executable
# -------------------------------------------------------------------
add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE yolov11_tensorrt)

# -------------------------------------------------------------------
# Copy the .so next to the executable so it's found at runtime
# -------------------------------------------------------------------
add_custom_command(TARGET demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/lib/libyolov11_tensorrt.so"
        "$<TARGET_FILE_DIR:demo>/"
    COMMENT "Copying libyolov11_tensorrt.so to build directory"
)

# RPATH: look for the .so in the same directory as the executable
set_target_properties(demo PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

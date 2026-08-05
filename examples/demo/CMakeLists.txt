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
# exec_detect_video – full-frame video detection
# -------------------------------------------------------------------
add_executable(exec_detect_video exec_detect_video.cpp)
target_link_libraries(exec_detect_video PRIVATE yolov11_tensorrt)

# -------------------------------------------------------------------
# exec_yolo_refine – tracking-guided ball refinement (YoloTrackRefine)
# -------------------------------------------------------------------
add_executable(exec_yolo_refine exec_yolo_refine.cpp)
target_link_libraries(exec_yolo_refine PRIVATE yolov11_tensorrt)

# -------------------------------------------------------------------
# Copy the .so next to the executables so it's found at runtime
# -------------------------------------------------------------------
add_custom_command(TARGET exec_detect_video POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/lib/libyolov11_tensorrt.so"
        "$<TARGET_FILE_DIR:exec_detect_video>/"
    COMMENT "Copying libyolov11_tensorrt.so to build directory"
)
add_custom_command(TARGET exec_yolo_refine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/lib/libyolov11_tensorrt.so"
        "$<TARGET_FILE_DIR:exec_yolo_refine>/"
    COMMENT "Copying libyolov11_tensorrt.so to build directory"
)

# RPATH: look for the .so in the same directory as the executable
set_target_properties(exec_detect_video PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
set_target_properties(exec_yolo_refine PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

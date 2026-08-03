/**
 * @file main.cpp
 * @brief Minimal C++ demo — uses libyolov11_tensorrt.so to run detection.
 *
 * Build:
 *   mkdir build && cd build && cmake .. && make -j
 *
 * Run (from project root):
 *   ./build/demo <engine> <video> [output.mp4] [output.json]
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "yolov11_api.h"

static void on_progress(int cur, int total, void* user) {
    (void)user;
    if (total > 0) {
        std::printf("\r  Progress: %d / %d  (%.1f%%)", cur, total, 100.0 * cur / total);
        std::fflush(stdout);
    } else {
        std::printf("\r  Progress: %d frames", cur);
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <engine> <video> [output.mp4] [output.json]\n",
                     argv[0]);
        return 1;
    }

    const char* engine   = argv[1];
    const char* video    = argv[2];
    const char* out_mp4  = (argc > 3) ? argv[3] : nullptr;
    const char* out_json = (argc > 4) ? argv[4] : nullptr;

    std::cout << "=== YOLOv11 TensorRT Demo ===" << std::endl;
    std::cout << "  engine:   " << engine << std::endl;
    std::cout << "  video:    " << video << std::endl;
    std::cout << "  out_mp4:  " << (out_mp4  ? out_mp4  : "(none)") << std::endl;
    std::cout << "  out_json: " << (out_json ? out_json : "(none)") << std::endl;
    std::cout << "==============================" << std::endl;

    auto t_start = std::chrono::steady_clock::now();

    yolov11_error_t ret = yolov11_detect_video(
        engine, video, out_mp4, out_json, on_progress, nullptr);

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << std::endl;

    if (ret == YOLOV11_OK) {
        std::cout << "Done." << std::endl;
    } else {
        std::cerr << "Error: " << ret << std::endl;
    }

    std::cout << "Total time: " << elapsed_ms / 1000.0 << " s  ("
              << elapsed_ms << " ms)" << std::endl;
    return (ret == YOLOV11_OK) ? 0 : 1;
}

#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "YOLOv11.h"


bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv)
{
    // Usage check
    if (argc < 2 || argc > 4) {
        printf("Usage:\n");
        printf("  Build engine:    %s <model.onnx>\n", argv[0]);
        printf("  Build && infer:  %s <model.onnx> <video_or_image>\n", argv[0]);
        printf("  Run inference:   %s <model.engine> <video_or_image> [output.mp4]\n", argv[0]);
        return 1;
    }

    // 统计最最开始的时间
    auto program_start_time = std::chrono::system_clock::now();
    const string engine_file_path{ argv[1] };

    // Optional output video path (3rd argument)
    string output_path;
    bool save_output = (argc >= 4);
    if (save_output) {
        output_path = argv[3];
    }

    // ---- Mode 1: Build engine only (1 argument, .onnx file) ----
    bool is_onnx = engine_file_path.find(".onnx") != std::string::npos;
    if (argc == 2) {
        if (!is_onnx) {
            printf("Error: single argument must be an .onnx file to build engine.\n");
            return 1;
        }
        printf("Building TensorRT engine from %s ...\n", engine_file_path.c_str());
        YOLOv11 model(engine_file_path, logger);
        printf("Engine built successfully.\n");
        printf("Program runtime: %.0f ms\n",
            (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now() - program_start_time).count());
        return 0;
    }

    // ---- Mode 2: Inference (2 arguments) ----
    const string path{ argv[2] };
    vector<string> imagePathList;
    bool isVideo{ false };

    if (IsFile(path))
    {
        string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm")
        {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path))
    {
        glob(path + "/*.jpg", imagePathList);
    }

    // Assume it's a folder, add logic to handle folders
    // init model
    YOLOv11 model(engine_file_path, logger);

    if (isVideo) {
        cv::VideoCapture cap(path);

        // Setup video writer if saving output
        cv::VideoWriter video_writer;
        if (save_output) {
            int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
            double out_fps = cap.get(cv::CAP_PROP_FPS);
            int out_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int out_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            video_writer.open(output_path, codec, out_fps, cv::Size(out_w, out_h));
            if (!video_writer.isOpened()) {
                printf("Error: Cannot open output video: %s\n", output_path.c_str());
                return 1;
            }
            printf("Saving output to: %s  (%dx%d @ %.1f FPS)\n",
                   output_path.c_str(), out_w, out_h, out_fps);
        }

        // ---- Dual-stream + batched inference ----
        double total_pre_ms = 0, total_inf_ms = 0, total_post_ms = 0;
        int frame_count = 0;
        const int B = model.getBatchSize();
        // Double-buffered images: 2 slots × B frames each
        vector<Mat> images_buf[2] = {vector<Mat>(B), vector<Mat>(B)};

        // --- Load first batch into slot 0 ---
        int cur_slot = 0;
        int cur_actual = 0;
        for (int b = 0; b < B; b++) {
            if (!cap.read(images_buf[cur_slot][b])) break;
            cur_actual++;
        }
        if (cur_actual == 0) { cap.release(); return 0; }

        // Preprocess + infer batch 0 on slot 0 (async — no host wait)
        auto t0 = std::chrono::system_clock::now();
        for (int b = 0; b < cur_actual; b++)
            model.preprocess(images_buf[cur_slot][b], cur_slot, b);
        auto t1 = std::chrono::system_clock::now();
        model.infer(cur_slot);
        auto t2 = std::chrono::system_clock::now();
        total_pre_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.;
        total_inf_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.;

        // --- Pipeline loop ---
        while (true) {
            int nxt_slot = 1 - cur_slot;

            // Read next batch
            int nxt_actual = 0;
            for (int b = 0; b < B; b++) {
                if (!cap.read(images_buf[nxt_slot][b])) break;
                nxt_actual++;
            }
            if (nxt_actual == 0) break;  // no more frames

            // Launch GPU work for next batch (async on other stream)
            auto pp0 = std::chrono::system_clock::now();
            for (int b = 0; b < nxt_actual; b++)
                model.preprocess(images_buf[nxt_slot][b], nxt_slot, b);
            auto pp1 = std::chrono::system_clock::now();
            model.infer(nxt_slot);
            auto pp2 = std::chrono::system_clock::now();
            total_pre_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(pp1 - pp0).count() / 1000.;
            total_inf_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(pp2 - pp1).count() / 1000.;

            // --- Finish current batch (GPU should be done by now) ---
            auto tp0 = std::chrono::system_clock::now();
            for (int b = 0; b < cur_actual; b++) {
                vector<Detection> objects;
                model.postprocess(objects, cur_slot, b);
                if (save_output) {
                    model.draw(images_buf[cur_slot][b], objects);
                    video_writer.write(images_buf[cur_slot][b]);
                }
            }
            auto tp1 = std::chrono::system_clock::now();
            total_post_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(tp1 - tp0).count() / 1000.;

            frame_count += cur_actual;
            cur_slot = nxt_slot;
            cur_actual = nxt_actual;
        }

        // --- Finish last batch ---
        {
            auto tp0 = std::chrono::system_clock::now();
            for (int b = 0; b < cur_actual; b++) {
                vector<Detection> objects;
                model.postprocess(objects, cur_slot, b);
                if (save_output) {
                    model.draw(images_buf[cur_slot][b], objects);
                    video_writer.write(images_buf[cur_slot][b]);
                }
            }
            auto tp1 = std::chrono::system_clock::now();
            total_post_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(tp1 - tp0).count() / 1000.;
            frame_count += cur_actual;
        }

        // Release resources
        cap.release();
        if (save_output) {
            video_writer.release();
            printf("Output saved to: %s\n", output_path.c_str());
        }

        printf("--- Dual-stream + Batch%d (%d frames) ---\n", B, frame_count);
        printf("  preprocess:  %.2f ms/frame\n", total_pre_ms / frame_count);
        printf("  inference:   %.2f ms/batch  (%.2f ms/frame)\n",
               total_inf_ms / (frame_count / B), total_inf_ms / frame_count);
        printf("  postprocess: %.2f ms/frame\n", total_post_ms / frame_count);
        printf("  total:       %.2f ms/frame  (%.1f FPS)\n",
               (total_pre_ms + total_inf_ms + total_post_ms) / frame_count,
               1000.0 * frame_count / (total_pre_ms + total_inf_ms + total_post_ms));
    }
    else{
        printf("not video\n");
        return 0;
    }

    auto program_end_time = std::chrono::system_clock::now();
    auto program_duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end_time - program_start_time).count();
    printf("Program runtime: %ld ms\n", program_duration);
    return 0;
}
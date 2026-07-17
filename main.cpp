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
        //path to video
        cv::VideoCapture cap(path);

        // Setup video writer if saving output
        cv::VideoWriter video_writer;
        if (save_output) {
            int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');  // H.264
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

        double total_pre_ms = 0, total_inf_ms = 0, total_post_ms = 0;
        int frame_count = 0;

        // Wall-clock timer for real end-to-end throughput
        auto inference_wall_start = std::chrono::system_clock::now();

        while (1)
        {
            Mat image;
            cap >> image;

            if (image.empty()) break;

            vector<Detection> objects;

            auto t0 = std::chrono::system_clock::now();
            model.preprocess(image);
            auto t1 = std::chrono::system_clock::now();
            model.infer();
            auto t2 = std::chrono::system_clock::now();
            model.postprocess(objects);
            auto t3 = std::chrono::system_clock::now();
            // 打印objects
            // for (const auto& obj : objects) {
            //     printf("class_id: %d, conf: %.2f, bbox: [%d, %d, %d, %d]\n",
            //         obj.class_id, obj.conf,
            //         obj.bbox.x, obj.bbox.y,
            //         obj.bbox.width, obj.bbox.height);
            // }

            // Save frame if output is enabled
            if (save_output) {
                model.draw(image, objects);
                video_writer.write(image);
            }

            auto pre_ms  = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.;
            auto inf_ms  = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.;
            auto post_ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.;
            total_pre_ms += pre_ms;
            total_inf_ms += inf_ms;
            total_post_ms += post_ms;
            frame_count++;
            // 保存图片
            // imwrite("output.jpg", image);
            // imshow("prediction", image);
            // waitKey(1);
        }

        // Release resources
        // destroyAllWindows();
        cap.release();
        if (save_output) {
            video_writer.release();
            printf("Output saved to: %s\n", output_path.c_str());
        }

        printf("--- Per-frame average (%d frames) ---\n", frame_count);
        printf("  preprocess:  %.2f ms\n", total_pre_ms / frame_count);
        printf("  inference:   %.2f ms\n", total_inf_ms / frame_count);
        printf("  postprocess: %.2f ms\n", total_post_ms / frame_count);
        printf("  pipeline:    %.2f ms  (%.1f GPU-pipeline FPS)\n",
               (total_pre_ms + total_inf_ms + total_post_ms) / frame_count,
               1000.0 * frame_count / (total_pre_ms + total_inf_ms + total_post_ms));

        auto inference_wall_end = std::chrono::system_clock::now();
        double wall_sec = std::chrono::duration<double>(inference_wall_end - inference_wall_start).count();
        printf("  real:        %.2f ms/frame  (%.1f end-to-end FPS, %.1fs wall clock)\n",
               1000.0 * wall_sec / frame_count, frame_count / wall_sec, wall_sec);
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
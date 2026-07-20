#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

class YOLOv11
{
public:
    static constexpr int NUM_STREAMS = 2;

    YOLOv11(string model_path, nvinfer1::ILogger& logger);
    ~YOLOv11();

    // ---- Pipeline API: slot = 0 or 1, batch_idx = 0..batch_size-1 ----
    void preprocess(Mat& image, int slot, int batch_idx = 0);
    void infer(int slot);
    void postprocess(vector<Detection>& output, int slot, int batch_idx = 0);
    void syncSlot(int slot);

    int getBatchSize() const { return batch_size; }    //!< Engine batch size

    // ---- Legacy single-stream API (backward compat) ----
    void preprocess(Mat& image) { preprocess(image, 0); }
    void infer()                 { infer(0); }
    void postprocess(vector<Detection>& output) { postprocess(output, 0); }

    void draw(Mat& image, const vector<Detection>& output);

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    // Double-buffered GPU resources (2 slots for pipelining)
    float* gpu_buffers[NUM_STREAMS][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    float* gpu_filtered_boxes[NUM_STREAMS] = {nullptr, nullptr};
    int*   gpu_filtered_count[NUM_STREAMS] = {nullptr, nullptr};
    float* cpu_filtered_boxes[NUM_STREAMS] = {nullptr, nullptr};

    cudaStream_t streams[NUM_STREAMS] = {nullptr, nullptr};
    cudaEvent_t  events[NUM_STREAMS]  = {nullptr, nullptr};
    IExecutionContext* contexts[NUM_STREAMS] = {nullptr, nullptr};

    bool inference_initialized = false;

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;

    // Model parameters
    int input_w = 0, input_h = 0;
    int batch_size = 1;           // Read from engine; 4 for batch export
    int num_detections = 0;       // 8400 or 300
    int detection_attribute_size = 0; // 84 or 6
    bool has_nms = false;        // true if engine has built-in NMS (det_attr <= 6)
    int num_classes = 80;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    const int MAX_OUTPUT_DETECTIONS = 1000;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;

    vector<Scalar> colors;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
};
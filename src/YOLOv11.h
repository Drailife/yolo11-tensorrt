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

    YOLOv11(string model_path, nvinfer1::ILogger& logger);
    ~YOLOv11();

    void preprocess(Mat& image);
    void infer();
    void postprocess(vector<Detection>& output);
    void draw(Mat& image, const vector<Detection>& output);

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    float* gpu_buffers[2] = {nullptr, nullptr};  //!< Device buffers for engine execution
    float* gpu_filtered_boxes = nullptr;          //!< GPU buffer for filtered detections
    int*   gpu_filtered_count = nullptr;          //!< GPU atomic counter for valid detections
    float* cpu_filtered_boxes = nullptr;          //!< CPU buffer for copying filtered results
    bool   inference_initialized = false;         //!< Whether init() has been called

    cudaStream_t stream = nullptr;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    // Model parameters
    int input_w;
    int input_h;
    int num_detections;
    int detection_attribute_size;
    int num_classes = 80;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    const int MAX_OUTPUT_DETECTIONS = 1000;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;

    vector<Scalar> colors;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);
};
#include "YOLOv11.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>


static Logger logger;
#define isFP16 true
#define warmup true


YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger)
{
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos)
    {
        init(model_path, logger);
    }
    // Build an engine from an onnx model
    else
    {
        build(model_path, logger);
        saveEngine(model_path);
    }

#if NV_TENSORRT_MAJOR < 10
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif
}


void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model
#if NV_TENSORRT_MAJOR < 10
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    auto output_dims = engine->getTensorShape(engine->getIOTensorName(1));
    detection_attribute_size = output_dims.d[2];  // 6 with NMS export
    num_detections = output_dims.d[1];            // 300 with NMS export
#endif

    // Initialize input buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer ([300, 6] = num_detections * detection_attribute_size)
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float)));

#if NV_TENSORRT_MAJOR >= 10
    context->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
    context->setOutputTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
#endif

    cuda_preprocess_init(MAX_IMAGE_SIZE);
    CUDA_CHECK(cudaStreamCreate(&stream));


    if (warmup) {
        for (int i = 0; i < 2; i++) {
            this->infer();
        }
        printf("model warmup 2 times\n");
    }

    inference_initialized = true;
}

YOLOv11::~YOLOv11()
{
    // Only release inference resources if init() was called
    if (inference_initialized) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        for (int i = 0; i < 2; i++)
            CUDA_CHECK(cudaFree(gpu_buffers[i]));
        cuda_preprocess_destroy();
    }

    // Destroy TensorRT objects (always created, both in build() and init())
    delete context;
    delete engine;
    delete runtime;
}

void YOLOv11::preprocess(Mat& image) {
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11::infer()
{
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    this->context->enqueueV3(this->stream);
#endif
}

void YOLOv11::postprocess(vector<Detection>& output)
{
    // Allocate CPU buffer for output [num_detections * detection_attribute_size]
    int output_size = num_detections * detection_attribute_size;
    float* cpu_output = new float[output_size];

    // Copy from GPU to CPU
    CUDA_CHECK(cudaMemcpyAsync(cpu_output, gpu_buffers[1],
                               output_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Parse NMS output: [num_detections, 6]  where 6 = [x1, y1, x2, y2, conf, class]
    for (int i = 0; i < num_detections; i++) {
        float x1   = cpu_output[i * 6 + 0];
        float y1   = cpu_output[i * 6 + 1];
        float x2   = cpu_output[i * 6 + 2];
        float y2   = cpu_output[i * 6 + 3];
        float conf = cpu_output[i * 6 + 4];
        int   cls  = (int)cpu_output[i * 6 + 5];

        // Skip invalid/padded entries
        if (conf <= 0.0f) continue;
        if (x1 == 0.0f && y1 == 0.0f && x2 == 0.0f && y2 == 0.0f) continue;

        Detection det;
        det.conf = conf;
        det.class_id = cls;
        // Convert xyxy to xywh (draw() expects xywh in model-input space)
        det.bbox = Rect(static_cast<int>(x1),
                        static_cast<int>(y1),
                        static_cast<int>(x2 - x1),
                        static_cast<int>(y2 - y1));
        output.push_back(det);
    }

    delete[] cpu_output;
}

void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    auto builder = createInferBuilder(logger);
#if NV_TENSORRT_MAJOR < 10
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
#else
    INetworkDefinition* network = builder->createNetworkV2(0U);
#endif
    IBuilderConfig* config = builder->createBuilderConfig();
    if (isFP16)
    {
#if NV_TENSORRT_MAJOR < 10
        config->setFlag(BuilderFlag::kFP16);
#else
        config->setFlag(BuilderFlag::kFP16);
#endif
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    runtime = createInferRuntime(logger);

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    context = engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool YOLOv11::saveEngine(const std::string& onnxpath)
{
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else
    {
        return false;
    }

    // Save the engine to the path
    if (engine)
    {
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}

void YOLOv11::draw(Mat& image, const vector<Detection>& output)
{
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}
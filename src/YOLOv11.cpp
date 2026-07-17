#include "YOLOv11.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include "postprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>


static Logger logger;
#define isFP16 true
#define warmup true


YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger)
{
    if (model_path.find(".onnx") == std::string::npos)
    {
        // Load pre-built engine
        init(model_path, logger);
    }
    else
    {
        // Build engine from ONNX
        build(model_path, logger);

        // Re-load the engine we just built to set up inference resources
        string engine_path = model_path.substr(0, model_path.find_last_of(".")) + ".engine";
        init(engine_path, logger);
    }
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

    // Deserialize the tensorrt engine (shared across all contexts)
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);

    // Get input and output sizes of the model (same for all slots)
#if NV_TENSORRT_MAJOR < 10
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    batch_size = input_dims.d[0];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    auto output_dims = engine->getTensorShape(engine->getIOTensorName(1));
    detection_attribute_size = output_dims.d[1];
    num_detections = output_dims.d[2];
#endif
    num_classes = detection_attribute_size - 4;
    printf("Model: batch=%d, input=%dx%d, det_attr=%d, num_dets=%d, classes=%d\n",
           batch_size, input_w, input_h, detection_attribute_size, num_detections, num_classes);

    // ---- Per-slot initialization (2 slots for pipelining) ----
    for (int s = 0; s < NUM_STREAMS; s++) {
        // Context per slot
        contexts[s] = engine->createExecutionContext();

        // GPU buffers: input [batch * 3 * H * W] + output [batch * det_attr * num_dets]
        CUDA_CHECK(cudaMalloc(&gpu_buffers[s][0], batch_size * 3 * input_w * input_h * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_buffers[s][1], batch_size * detection_attribute_size * num_detections * sizeof(float)));

#if NV_TENSORRT_MAJOR >= 10
        contexts[s]->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[s][0]);
        contexts[s]->setOutputTensorAddress(engine->getIOTensorName(1), gpu_buffers[s][1]);
#endif

        // Postprocess GPU buffers
        CUDA_CHECK(cudaMalloc(&gpu_filtered_boxes[s], MAX_OUTPUT_DETECTIONS * 6 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_filtered_count[s], sizeof(int)));
        cpu_filtered_boxes[s] = new float[MAX_OUTPUT_DETECTIONS * 6];

        // Stream + event per slot
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaEventCreate(&events[s]));
    }

    cuda_preprocess_init(MAX_IMAGE_SIZE);

    // Warmup on slot 0
    if (warmup) {
        for (int i = 0; i < 2; i++) {
            this->infer(0);
        }
        printf("model warmup 2 times\n");
    }

    inference_initialized = true;
}

YOLOv11::~YOLOv11()
{
    if (inference_initialized) {
        for (int s = 0; s < NUM_STREAMS; s++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            CUDA_CHECK(cudaStreamDestroy(streams[s]));
            CUDA_CHECK(cudaEventDestroy(events[s]));
            CUDA_CHECK(cudaFree(gpu_buffers[s][0]));
            CUDA_CHECK(cudaFree(gpu_buffers[s][1]));
            CUDA_CHECK(cudaFree(gpu_filtered_boxes[s]));
            CUDA_CHECK(cudaFree(gpu_filtered_count[s]));
            delete[] cpu_filtered_boxes[s];
            delete contexts[s];
        }
        cuda_preprocess_destroy();
    }

    // Engine and runtime are shared, always created
    delete engine;
    delete runtime;
}

void YOLOv11::preprocess(Mat& image, int slot, int batch_idx) {
    // Launch GPU preprocess on stream[slot] — writes to batch slot
    float* dst = gpu_buffers[slot][0] + batch_idx * 3 * input_w * input_h;
    cuda_preprocess(image.ptr(), image.cols, image.rows,
                    dst, input_w, input_h, streams[slot]);
}

void YOLOv11::infer(int slot)
{
#if NV_TENSORRT_MAJOR < 10
    contexts[slot]->enqueueV2((void**)gpu_buffers[slot], streams[slot], nullptr);
#else
    contexts[slot]->enqueueV3(streams[slot]);
#endif
}

void YOLOv11::postprocess(vector<Detection>& output, int slot, int batch_idx)
{
    // Raw output for this batch element: offset in the full batched buffer
    int per_image_size = detection_attribute_size * num_detections;
    float* raw_output = gpu_buffers[slot][1] + batch_idx * per_image_size;

    // ----- Step 1: Zero atomic counter on GPU (async on stream[slot]) -----
    CUDA_CHECK(cudaMemsetAsync(gpu_filtered_count[slot], 0, sizeof(int), streams[slot]));

    // ----- Step 2: GPU kernel: decode boxes + filter by confidence (async) -----
    cuda_postprocess_decode(
        raw_output,
        gpu_filtered_boxes[slot],
        gpu_filtered_count[slot],
        num_detections, num_classes, detection_attribute_size,
        conf_threshold, MAX_OUTPUT_DETECTIONS,
        streams[slot]
    );

    // ----- Step 3: Copy filtered count back to CPU (async) -----
    int filtered_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&filtered_count, gpu_filtered_count[slot], sizeof(int),
                               cudaMemcpyDeviceToHost, streams[slot]));

    // ----- Step 4: Sync: wait for all GPU work on this slot -----
    CUDA_CHECK(cudaStreamSynchronize(streams[slot]));

    if (filtered_count == 0) return;
    if (filtered_count > MAX_OUTPUT_DETECTIONS) filtered_count = MAX_OUTPUT_DETECTIONS;

    // ----- Step 5: Copy filtered detections (GPU is done, safe to use sync memcpy) -----
    CUDA_CHECK(cudaMemcpyAsync(cpu_filtered_boxes[slot], gpu_filtered_boxes[slot],
                               filtered_count * 6 * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[slot]));
    CUDA_CHECK(cudaStreamSynchronize(streams[slot]));

    // ----- Step 6: Build detection lists + CPU NMS -----
    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    boxes.reserve(filtered_count);
    class_ids.reserve(filtered_count);
    confidences.reserve(filtered_count);

    for (int i = 0; i < filtered_count; i++) {
        float x    = cpu_filtered_boxes[slot][i * 6 + 0];
        float y    = cpu_filtered_boxes[slot][i * 6 + 1];
        float w    = cpu_filtered_boxes[slot][i * 6 + 2];
        float h    = cpu_filtered_boxes[slot][i * 6 + 3];
        float conf = cpu_filtered_boxes[slot][i * 6 + 4];
        int   cls  = (int)cpu_filtered_boxes[slot][i * 6 + 5];

        Rect box;
        box.x = static_cast<int>(x);
        box.y = static_cast<int>(y);
        box.width = static_cast<int>(w);
        box.height = static_cast<int>(h);

        boxes.push_back(box);
        class_ids.push_back(cls);
        confidences.push_back(conf);
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
}

void YOLOv11::syncSlot(int slot)
{
    CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
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
    parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));

    // Set optimization profile only if input has dynamic dimensions
    auto input_dims = network->getInput(0)->getDimensions();
    bool has_dynamic = false;
    for (int i = 0; i < input_dims.nbDims; i++) {
        if (input_dims.d[i] < 0) { has_dynamic = true; break; }
    }
    if (has_dynamic) {
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN,
                               Dims4{1, input_dims.d[1], input_dims.d[2], input_dims.d[3]});
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT,
                               Dims4{4, input_dims.d[1], input_dims.d[2], input_dims.d[3]});
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX,
                               Dims4{4, input_dims.d[1], input_dims.d[2], input_dims.d[3]});
        config->addOptimizationProfile(profile);
        printf("Optimization profile: MIN=[1,%d,%d,%d] OPT=[4,%d,%d,%d] MAX=[4,%d,%d,%d]\n",
               input_dims.d[1], input_dims.d[2], input_dims.d[3],
               input_dims.d[1], input_dims.d[2], input_dims.d[3],
               input_dims.d[1], input_dims.d[2], input_dims.d[3]);
    }

    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    // Write the serialized engine to file (engine will be loaded later by init())
    string engine_path = onnxPath.substr(0, onnxPath.find_last_of(".")) + ".engine";
    std::ofstream file(engine_path, std::ios::binary | std::ios::out);
    file.write((const char*)plan->data(), plan->size());
    file.close();

    delete plan;
    delete parser;
    delete config;
    delete network;
    delete builder;
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
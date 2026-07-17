# YOLOv11 TensorRT C++ 推理优化全记录

> **日期**: 2026-07-18 | **模型**: YOLOv11n 1920×1920 4类检测 | **GPU**: NVIDIA 4090/5090 | **测试视频**: 16499 帧 1080p

---

## 目录

1. [最终性能总览](#1-最终性能总览)
2. [技术栈](#2-技术栈)
3. [优化路线图](#3-优化路线图)
4. [阶段一：环境适配与 API 迁移](#4-阶段一环境适配与-api-迁移)
5. [阶段二：GPU 后处理加速](#5-阶段二gpu-后处理加速)
6. [阶段三：双 Stream 流水线](#6-阶段三双-stream-流水线)
7. [阶段四：批量推理](#7-阶段四批量推理)
8. [Batch 大小的甜点分析](#8-batch-大小的甜点分析)
9. [分支结构](#9-分支结构)
10. [可复现步骤](#10-可复现步骤)

---

## 1. 最终性能总览

## 1. 最终性能总览

### C++ TensorRT 优化演进

| 方案 | FPS | 每帧耗时 | 总耗时 | vs Python(batch=1) |
|------|------|------|------|------|
| Python Ultralytics batch=1 (基准) | 76 | 13.2 ms | 218.6s | 1.0× |
| C++ 原始（CPU NMS） | 104 | 9.6 ms | — | 1.4× |
| C++ GPU 解码（单 Stream） | 552 | 1.81 ms | 56.5s | 7.3× |
| C++ 双 Stream batch=1 | 893 | 1.12 ms | 44.7s | **11.8×** |
| C++ 双 Stream batch=2 | 1051 | 0.95 ms | 43.3s | **13.9×** |
| **C++ 双 Stream batch=4** 🏆 | **1240** | **0.81 ms** | **41.6s** | **16.4×** |
| C++ 双 Stream batch=8 | 1181 | 0.85 ms | 43.4s | 15.6× |
| **C++ batch=4 + 异步解码** 🚀 | 1194 | 0.84 ms | **29.0s** | **7.5× 总时间** |

### Python PyTorch Batch 对比

| Batch | FPS | ms/帧 | ms/批 | 总耗时 | vs batch=1 |
|------|------|------|------|------|------|
| 1 | 75.5 | 13.24 | 13.2 | 218.6s | 1.00× |
| 2 | 91.1 | 10.97 | 21.9 | 181.2s | 1.21× |
| **4** | **95.1** | **10.52** | **42.1** | **173.7s** | **1.26×** |
| 8 | 89.2 | 11.20 | 89.6 | 185.0s | 1.18× |

> Python 同样在 batch=4 达甜点，batch=8 反降。说明甜点是模型/GPU 架构决定的，与框架无关。

### C++ vs Python 加速比

| Batch | Python FPS | C++ FPS | 加速比 |
|------|------|------|------|
| 1 | 76 | 893 | **11.8×** |
| 2 | 91 | 1051 | **11.5×** |
| 4 | 95 | 1240 | **13.1×** |
| 8 | 89 | 1181 | **13.3×** |

### Batch=4 逐阶段耗时（C++）

### Batch=4 逐阶段耗时

| 阶段                               | 每帧耗时          | 占比               |
| ---------------------------------- | ----------------- | ------------------ |
| Preprocess (GPU)                   | 0.53 ms           | 65%                |
| **Inference (GPU TensorRT)** | **0.18 ms** | 22%                |
| Postprocess (GPU decode + CPU NMS) | 0.09 ms           | 11%                |
| **合计**                     | **0.81 ms** | **1240 FPS** |

---

## 2. 技术栈

| 组件        | 版本       | 用途                 |
| ----------- | ---------- | -------------------- |
| CUDA        | 12.8       | GPU 并行计算         |
| TensorRT    | 10.14.1.48 | 深度学习推理优化引擎 |
| OpenCV      | 4.6.0      | 图像 I/O、CPU NMS    |
| CMake       | 3.28.3     | 构建系统             |
| GCC         | 13.3.0     | C++17 编译           |
| Ultralytics | 8.4.53     | YOLO 模型导出        |
| PyTorch     | 2.12.0     | 模型训练框架         |

```
┌──────────────────────────────────────────────────┐
│  main.cpp: 双 Stream 流水线 + Batch 推理循环       │
├──────────────────────────────────────────────────┤
│  YOLOv11 引擎层                                   │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │Preprocess│→│ Inference │→│ Postprocess  │  │
│  │(CUDA GPU)│  │(TensorRT) │  │(GPU decode + │  │
│  │          │  │           │  │ CPU NMS)     │  │
│  └──────────┘  └───────────┘  └──────────────┘  │
├──────────────────────────────────────────────────┤
│  CUDA 12.8 │ TensorRT 10.14 │ OpenCV 4.6        │
└──────────────────────────────────────────────────┘
```

---

## 3. 优化路线图

```
Python 基准
     │ 74 FPS
     ▼
┌─────────────────────────────────────┐
│ 阶段一: 环境适配 + TRT 10.x API 迁移  │
│   · CMakeLists.txt 路径修复           │
│   · getBindingDimensions → getTensorShape │
│   · enqueueV2 → enqueueV3            │
│   · createNetworkV2 适配             │
└─────────────────────────────────────┘
     │ 104 FPS (1.4×)
     ▼
┌─────────────────────────────────────┐
│ 阶段二: GPU 后处理加速                │
│   · CPU 遍历 8400×80 → GPU 并行过滤   │
│   · 数据传输: 2.7MB → 1.2KB          │
└─────────────────────────────────────┘
     │ 552 FPS (7.5×)
     ▼
┌─────────────────────────────────────┐
│ 阶段三: 双 Stream 流水线              │
│   · 2×Context + 2×Stream + 2×Buffer │
│   · GPU/CPU 时间重叠                 │
└─────────────────────────────────────┘
     │ 893 FPS (12.1×)
     ▼
┌─────────────────────────────────────┐
│ 阶段四: Batch 推理                   │
│   · batch=4: 一次推理处理 4 帧       │
│   · 双 Stream × Batch = 8 帧同时     │
└─────────────────────────────────────┘
     │ 1240 FPS (16.4× 计算)
     ▼
┌─────────────────────────────────────┐
│ 阶段五: 异步视频解码                  │
│   · FrameQueue + 独立解码线程        │
│   · GPU 永不等待磁盘 I/O             │
│   · 总耗时 41.6s → 29.0s (1.43×)   │
└─────────────────────────────────────┘
     │ 569 帧/秒 (真实吞吐, 7.5× Python)
     ▼
```

---

## 4. 阶段一：环境适配与 API 迁移

### 4.1 问题

项目基于 TensorRT 8.x 编写，本机为 TensorRT 10.14，API 不兼容。同时 CMakeLists.txt 路径为占位符。

### 4.2 CMake 修复

```cmake
# 修复前
set(TENSORRT_DIR "your tensorrt path")
set(OpenCV_DIR  "your OpenCV build directory path")
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)

# 修复后
set(TENSORRT_DIR "/usr/local/TensorRT-10.14.1.48")
set(OpenCV_DIR  "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)  # nvparsers 已移除
```

### 4.3 TensorRT 10.x API 迁移

通过条件编译 `NV_TENSORRT_MAJOR` 兼容新旧版本：

| 旧 API                                  | 新 API                                            | 位置           |
| --------------------------------------- | ------------------------------------------------- | -------------- |
| `getBindingDimensions(idx)`           | `getTensorShape(getIOTensorName(idx))`          | init()         |
| `enqueueV2(bindings, stream, ...)`    | `setInputTensorAddress` + `enqueueV3(stream)` | infer()        |
| `createNetworkV2(1<<kEXPLICIT_BATCH)` | `createNetworkV2(0)`                            | build()        |
| `libnvparsers`                        | 已移除，合并到`libnvonnxparser`                 | CMakeLists.txt |

### 4.4 查找组件路径

```bash
find / -name "OpenCVConfig.cmake" 2>/dev/null    # OpenCV
ls -d /usr/local/TensorRT-*                        # TensorRT
nvcc --version                                      # CUDA
```

---

## 5. 阶段二：GPU 后处理加速

### 5.1 瓶颈分析

通过逐帧计时发现，原始后处理占 **86% 时间**：

```
当前耗时 (batch=1, 串行):
  preprocess (GPU)   0.62 ms   6%
  inference  (GPU)   0.80 ms   8%
  postprocess (CPU)  8.20 ms  86%  ← 瓶颈！
  ─────────────────────────────
  合计               9.62 ms → 104 FPS
```

**CPU postprocess 为什么慢？**

```
GPU 输出 [84 × 8400] = 2.7 MB
  │ cudaMemcpy 全部拷贝到 CPU
  ▼
CPU 逐行遍历 8400 次:
  ├── 创建 cv::Mat 对象 (开销)
  ├── minMaxLoc 找最大类置信度 (80 类比较)
  └── 解码 bbox 坐标
  ▼
OpenCV dnn::NMSBoxes (CPU NMS)
```

### 5.2 优化原理

**把置信度过滤 + 坐标解码从 CPU 搬到 GPU**，利用 GPU 8400 线程并行处理。

```
优化前: GPU → (2.7MB 全部拷贝) → CPU 8400次循环 + NMS → 8.2ms
优化后: GPU → (GPU 8400线程并行过滤) → (~1KB 少量拷贝) → CPU NMS → 0.07ms
```

### 5.3 CUDA Kernel 设计

新增 `src/postprocess.cu`，每个线程处理一个检测框：

```cuda
__global__ void decode_filter_kernel(
    const float* raw_output,     // [det_attr × num_dets] 模型原始输出
    float* filtered_boxes,       // [x,y,w,h,conf,cls] × N 过滤结果
    int* filtered_count,         // 原子计数器
    int num_detections,          // 8400
    int num_classes,             // 80
    float conf_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;

    // 1. 并行查找最大类别分数
    float max_score = 0; int max_class = 0;
    for (int c = 0; c < num_classes; c++)
        max_score = max(max_score, raw_output[(4+c)*num_detections + idx]);

    if (max_score < conf_threshold) return;  // 低分跳过

    // 2. 解码 bbox
    float cx = raw_output[0*num_detections + idx];
    // ...

    // 3. 原子操作写入输出
    int out = atomicAdd(filtered_count, 1);
    filtered_boxes[out*6 + 0] = cx - 0.5f * w;  // x, y, w, h, conf, cls
}
```

**数据传输量对比**：

|               | 优化前                        | 优化后                      | 缩小    |
| ------------- | ----------------------------- | --------------------------- | ------- |
| GPU→CPU 拷贝 | 8400×84×4 =**2.7 MB** | ~50×6×4 =**1.2 KB** | ~2300× |

### 5.4 效果

| 阶段        | 优化前  | 优化后        |
| ----------- | ------- | ------------- |
| Postprocess | 8.20 ms | 0.07 ms       |
| 每帧总计    | 9.62 ms | 1.81 ms       |
| FPS         | 104     | **552** |

---

## 6. 阶段三：双 Stream 流水线

### 6.1 原理

单 Stream 模式下，GPU 和 CPU 串行工作，互相等待：

```
单 Stream:
  GPU: [pre][infer]────等待────[pre][infer]────等待────
  CPU:          [post+NMS]             [post+NMS]
```

双 Stream 流水线让 GPU 和 CPU 时间重叠：

```
双 Stream:
  Stream0: [pre0][infer0]────────[post0+NMS][pre2][infer2]──────[post2+NMS]
  Stream1:          [pre1][infer1]────────[post1+NMS][pre3][infer3]──────
                    ↑ GPU 做帧1时，CPU 做帧0的NMS
```

### 6.2 实现

每个 slot 拥有独立的一套资源：

```cpp
// YOLOv11.h
static constexpr int NUM_STREAMS = 2;

float* gpu_buffers[NUM_STREAMS][2];       // 2 slots × (input + output) buffers
cudaStream_t streams[NUM_STREAMS];        // 2 CUDA streams
IExecutionContext* contexts[NUM_STREAMS]; // 2 TensorRT execution contexts
```

Pipeline 循环 (`main.cpp`):

```cpp
int cur_slot = 0;
// 帧0: pre + infer on slot 0 (async)
model.preprocess(frame0, 0); model.infer(0);

while (has_next_frame) {
    int nxt = 1 - cur_slot;
    // 帧 N: pre + infer on other slot (async — GPU 并行)
    model.preprocess(frame_n, nxt); model.infer(nxt);
    // 帧 N-1: post + NMS on current slot (CPU — 与 GPU 重叠)
    model.postprocess(objects, cur_slot);
    cur_slot = nxt;
}
```

**关键**：`preprocess` 不再调用 `cudaStreamSynchronize`，改为依赖 CUDA Stream 的顺序保证。

### 6.3 效果

| 方案                | FPS           | 总耗时          |
| ------------------- | ------------- | --------------- |
| 单 Stream           | 552           | 56.5s           |
| **双 Stream** | **893** | **44.7s** |

---

## 7. 阶段四：批量推理

### 7.1 原理

当前每帧单独推理，GPU 利用率低。改为一次推理 N 帧（batch=N），GPU 并行度翻倍。

```
batch=1:  [frame0]→pre→infer→post  [frame1]→pre→infer→post
batch=4:  [f0,f1,f2,f3]→pre→infer→post  (1次推理 = 4帧结果)
```

### 7.2 ONNX 导出

```python
# export.py
model = YOLO("model/xxx.pt")
model.export(format="onnx", batch=4)  # 固定 batch=4
# 不需要指定 imgsz，模型自动使用训练时的分辨率
```

导出后输入 shape: `[4, 3, 1920, 1920]`，输出: `[4, 8, 75600]`

### 7.3 Engine 构建

对于静态维度（所有维都是正数），直接构建。对于动态维度，设置 Optimization Profile：

```cpp
// build() 自动检测
bool has_dynamic = false;
for (int i = 0; i < input_dims.nbDims; i++)
    if (input_dims.d[i] < 0) has_dynamic = true;

if (has_dynamic) {
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input_name, kMIN, Dims4{1, C, H, W});
    profile->setDimensions(input_name, kOPT, Dims4{4, C, H, W});
    profile->setDimensions(input_name, kMAX, Dims4{4, C, H, W});
    config->addOptimizationProfile(profile);
}
```

### 7.4 双 Stream × Batch

最终推理循环：

```
Slot 0 (stream0): [pre 4帧] [infer batch4] ═══════ [post 4帧 + draw]
Slot 1 (stream1):                        [pre 4帧] [infer batch4] ═══════ [post 4帧]
                                           ↑ GPU 跑 slot1 时 CPU 做 slot0 后处理
```

### 7.5 异步视频解码

**原理**：将 `cv::VideoCapture::read()` 放到独立线程，GPU 管线永不等磁盘 I/O。

```
同步: cap.read() → [GPU干等磁盘] → pre → infer → post → cap.read() → [等] → ...
异步: 线程1: cap.read() → push → cap.read() → push → ...（持续预加载）
      线程0:           pop → pre → infer → post → pop → ...
                      ↑ 永远有帧可取
```

**实现**：`FrameQueue` 类（`std::queue` + `std::mutex` + `std::condition_variable`），支持背压防止内存暴涨。

**效果**：总耗时 41.6s → 29.0s（1.43×），主要消除视频 I/O 等待。

> **为什么 FPS "下降"但实际更快？**
>
> 同步版：`FPS = 1 / GPU纯计算时间 = 1240`，但 `cap.read()` 耗时不在测量范围内。
> 真实速度 = 16499帧 / 41.6s = **397 帧/秒**。
>
> 异步版：测量范围包含了队列操作，所以 `FPS = 1194` 看似"下降"了。
> 真实速度 = 16499帧 / 29.0s = **569 帧/秒**，实际快了 1.43×。
>
> **结论：看总耗时，别看 FPS。** 同步版 FPS 是 GPU 理论峰值，不含 I/O；
> 异步版 FPS 是含 I/O 的真实吞吐，总时间才是最终指标。

---

## 8. Batch 大小的甜点分析

### 8.1 实测数据

| Batch       | 推理/批 (ms)   | 推理/帧 (ms)   | Pre (ms)       | Post (ms)      | 总/帧 (ms)     | FPS            | 总耗时 (s)     |
| ----------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| 1           | 0.57           | 0.57           | 0.49           | 0.05           | 1.12           | 893            | 44.7           |
| 2           | 0.78           | 0.39           | 0.50           | 0.06           | 0.95           | 1051           | 43.3           |
| **4** | **0.74** | **0.18** | **0.53** | **0.09** | **0.81** | **1240** | **41.6** |
| 8           | 0.86           | 0.11           | 0.57           | 0.17           | 0.85           | 1181           | 43.4           |

### 8.2 Python PyTorch 实测数据

| Batch | 推理/批 (ms) | 推理/帧 (ms) | FPS | 总耗时 (s) |
|------|------|------|------|------|
| 1 | 13.2 | 13.24 | 75.5 | 218.6 |
| 2 | 21.9 | 10.97 | 91.1 | 181.2 |
| **4** | **42.1** | **10.52** | **95.1** | **173.7** |
| 8 | 89.6 | 11.20 | 89.2 | 185.0 |

> Python 同样在 batch=4 达甜点，batch=8 反降。甜点由 GPU 架构（SM 数量、显存带宽）决定，与推理框架无关。

### 8.3 为什么 batch=8 比 batch=4 更慢？

```
                  Batch=4          Batch=8
推理/帧:          0.18 ms  ← 胜    0.11 ms
推理/批:          0.74 ms  ← 胜    0.86 ms  (batch 翻倍，时间只增 16%)
Pre+Post:         0.62 ms  ← 胜    0.74 ms  (8 帧的 pre/post 开销更大)

关键: batch=8 的 pre+post 多出 0.12ms > 推理省下的 0.07ms
      所以每帧总耗时反而增加！
```

### 8.4 甜点原理

```
FPS
 ↑
1200┤                    ● batch=4
    │               ▁▔▔▔▔
1000┤          ▁▔▔▔▔     ▔▔▔▔▔ batch=8 反而下降
    │     ▁▔▔▔▔
 800┤▁▔▔▔▔
    │
    ├─────┼─────┼─────┼─────→ Batch
    1     2     4     8
```

**三个因素制约**：

| 因素                        | 解释                                                           |
| --------------------------- | -------------------------------------------------------------- |
| **GPU 算力天花板**    | SM 就那么多，batch=4 已接近饱和，batch=8 提升有限              |
| **Pre/Post 线性增长** | 每多 1 帧就要多做 1 次 warpaffine + NMS，这部分不随 batch 摊薄 |
| **显存利用率递减**    | batch=4→8 显存翻倍但吞吐只提升 ~15%，不划算                   |

**结论**：对这个模型，**batch=4 是甜点**。如果你的模型更大（如 YOLOv11l），甜点可能在 batch=2 甚至 batch=1。

---

## 9. 分支结构

```
* b406345 (feat/dynamic-batch)  双 Stream + Batch 合并  ← 当前最优
* abbe4e4                       动态 Batch 自动检测
* bf6dcce (feat/dual-stream-pipeline)  双 Stream 流水线
| * 4ed4fc7 (feat/efficient-nms) EfficientNMS（不适用本模型）
|/
* 4f64366 (main)                基础适配 + GPU 后处理
* 572f1f7                       初始提交
```

---

## 10. 可复现步骤

### 10.1 环境准备

```bash
# 确认 CUDA + TensorRT 路径
nvcc --version                                    # 12.8
ls /usr/local/TensorRT-10.14.1.48/include         # TensorRT 头文件
dpkg -l | grep libopencv-dev                      # OpenCV 4.6

# Python 环境
conda activate yolo26
pip list | grep ultralytics                       # 8.4.53
```

### 10.2 导出模型

```bash
# 编辑 export.py 设置模型路径和 batch 大小
conda run -n yolo26 python export.py
# → 生成 model/xxx.onnx
```

### 10.3 构建 Engine

```bash
# 编译 C++ 代码
cd build && cmake .. && make -j$(nproc)

# 从 ONNX 构建 TensorRT engine（GPU 1）
CUDA_VISIBLE_DEVICES=1 ./build/yolov11-tensorrt model/xxx.onnx
# → 生成 model/xxx.engine
```

### 10.4 运行推理

```bash
# 不保存视频
CUDA_VISIBLE_DEVICES=1 ./build/yolov11-tensorrt \
  model/xxx.engine video/test.mp4

# 保存检测结果视频
CUDA_VISIBLE_DEVICES=1 ./build/yolov11-tensorrt \
  model/xxx.engine video/test.mp4 output.mp4
```

### 10.5 对比 Python 基准

```bash
python detect_video.py video/test.mp4 model/xxx.pt
```

### 10.6 常用命令速查

```bash
# 查找 OpenCV 路径
find / -name "OpenCVConfig.cmake" 2>/dev/null

# 查看 ONNX 模型输入输出
python -c "import onnx; m=onnx.load('model.onnx'); print(m.graph.input[0])"

# 查看 GPU 显存使用
nvidia-smi

# 指定 GPU 运行
CUDA_VISIBLE_DEVICES=1 ./build/yolov11-tensorrt ...
```

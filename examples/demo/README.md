# YOLOv11 TensorRT Demo

基于 `libyolov11_tensorrt.so` 的 C++ 示例程序集，演示 C API 的两种使用场景。

## 可执行文件

| 可执行文件            | 源文件                    | 功能                   |
| --------------------- | ------------------------- | ---------------------- |
| `exec_detect_video` | `exec_detect_video.cpp` | 全帧视频目标检测       |
| `exec_yolo_refine`  | `exec_yolo_refine.cpp`  | 跟踪引导的篮球检测细化 |

---

## 一、`exec_detect_video` — 全帧视频目标检测

对视频逐帧（或批量）运行 YOLO 推理，输出检测结果 JSON 和/或标注视频。

### 用法

```bash
./build/exec_detect_video <engine> <video> [output.json] [output.mp4] [conf]
```

| 参数            | 说明                                                             |
| --------------- | ---------------------------------------------------------------- |
| `engine`      | TensorRT engine 路径（`.engine`），也可传入 `.onnx` 自动构建 |
| `video`       | 输入视频路径                                                     |
| `output.json` | （可选）输出 JSON 路径，`None` 跳过                            |
| `output.mp4`  | （可选）输出标注视频路径，`None` 跳过                          |
| `conf`        | （可选）置信度阈值 (0.0~1.0)，默认 0.3                           |

### 示例

```bash
# 仅推理，不输出文件
./build/exec_detect_video model.engine video.mp4

# 输出 JSON + 标注视频
./build/exec_detect_video model.engine video.mp4 result.json output.mp4

# 跳过视频输出，只输出 JSON，置信度 0.5
./build/exec_detect_video model.engine video.mp4 result.json None 0.5
```

---

## 二、`exec_yolo_refine` — 跟踪引导的篮球检测细化（YoloTrackRefine）

读取流水线第一阶段的全帧 YOLO 检测结果（`JsonForLLM_with_objects.json`），
利用跟踪 + 速度预测 + 裁剪重检测，补回漏检的篮球。

> 详细算法说明见 [`doc/yolo-track-refine.md`](../../doc/yolo-track-refine.md)。

### 用法

```bash
./build/exec_yolo_refine <json> <video> <engine> [output.json] [options]
```

| 参数            | 说明                                                              |
| --------------- | ----------------------------------------------------------------- |
| `json`        | `JsonForLLM_with_objects.json` 路径                             |
| `video`       | 输入视频路径                                                      |
| `engine`      | 640×640 球检测 TensorRT 引擎路径                                 |
| `output.json` | （可选）输出路径，默认`<json_dir>/YoloRefineBallboxes_cpp.json` |

| 选项                  | 默认值 | 说明                      |
| --------------------- | ------ | ------------------------- |
| `--stride N`        | 2      | 帧步长                    |
| `--conf C`          | 0.25   | 裁剪重检测置信度阈值      |
| `--lookback N`      | 5      | 最大回溯帧数              |
| `--proximity P`     | 150    | 中心距离阈值（像素）      |
| `--ball-cls N`      | 1      | 球检测模型中篮球的类别 ID |
| `--json-ball-cls N` | 1      | 输入 JSON 中篮球的类别 ID |

### 示例

```bash
./build/exec_yolo_refine \
  JsonForLLM_with_objects.json \
  video.mp4 \
  ball_detect_640.engine \
  --stride 1 --conf 0.25 --ball-cls 0 --json-ball-cls 1
```

---

## 构建

### 1. 编译 `libyolov11_tensorrt.so`

在项目根目录下执行：

```bash
mkdir -p build && cd build
cmake .. && make -j
```

编译产物 `libyolov11_tensorrt.so` 会生成在 `build/` 目录下，将其复制到 `examples/demo/lib/`：

```bash
cp build/libyolov11_tensorrt.so examples/demo/lib/
```

> **前置依赖**：CUDA、TensorRT（默认 `/usr/local/TensorRT-10.14.1.48`）、OpenCV。

### 2. 编译 demo 可执行文件

```bash
cd examples/demo
mkdir -p build && cd build
cmake .. && make -j
```

产物：

```
build/
  exec_detect_video          ← 全帧检测
  exec_yolo_refine           ← 跟踪细化
```

## 依赖

| 依赖                       | 版本                 | 说明                                        |
| -------------------------- | -------------------- | ------------------------------------------- |
| CMake                      | >= 3.12              | 构建系统                                    |
| C++17                      | -                    | 编译器需支持 C++17                          |
| CUDA                       | -                    | GPU 加速                                    |
| TensorRT                   | **10.14.1.48** | `/usr/local/TensorRT-10.14.1.48`          |
| OpenCV                     | **4.6.0**      | `/usr/lib/x86_64-linux-gnu/cmake/opencv4` |
| `libyolov11_tensorrt.so` | -                    | 本项目编译生成，放置于`lib/` 目录         |

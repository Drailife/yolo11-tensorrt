# YOLOv11 TensorRT Demo

基于 `libyolov11_tensorrt.so` 的最小化 C++ 示例程序，演示如何调用 C API 进行视频目标检测。

## 功能

- 加载 TensorRT engine（`.engine`）运行推理
- 支持直接传入 `.onnx` 自动构建 engine
- 可选输出带标注的视频（`.mp4`）
- 可选输出逐帧检测结果的 JSON 文件
- 支持进度回调

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

> **前置依赖**：需要安装 CUDA、TensorRT（默认路径 `/usr/local/TensorRT-10.14.1.48`）和 OpenCV。如果 TensorRT 路径不同，请修改根目录 `CMakeLists.txt` 中的 `TENSORRT_DIR`。

### 2. 编译 demo

```bash
cd examples/demo
mkdir -p build && cd build
cmake .. && make -j
```

## 运行

```bash
./build/demo <engine> <video> [output.json] [output.mp4] [conf]
```

| 参数 | 说明 |
|------|------|
| `engine` | TensorRT engine 文件路径（`.engine`），也可传入 `.onnx` 自动构建 |
| `video` | 输入视频路径 |
| `output.json` | （可选）输出检测结果 JSON 路径，传入 `None` 跳过 |
| `output.mp4` | （可选）输出标注视频路径，传入 `None` 跳过 |
| `conf` | （可选）置信度阈值 (0.0~1.0)，默认 0.3 |

### 示例

```bash
# 仅推理，不输出文件
./build/demo ../../model/yolo11n_with_nms.engine ../../video/test.mp4

# 输出 JSON + 标注视频（跳过不需要的用 None）
./build/demo ../../model/yolo11n_with_nms.engine ../../video/test.mp4 result.json None

# 指定置信度阈值 0.5，输出 JSON
./build/demo ../../model/yolo11n_with_nms.engine ../../video/test.mp4 result.json None 0.5
```

## 依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| CMake | >= 3.12 | 构建系统 |
| C++17 | - | 编译器需支持 C++17 |
| CUDA | - | GPU 加速 |
| TensorRT | **10.14.1.48** | NVIDIA 推理优化引擎，默认路径 `/usr/local/TensorRT-10.14.1.48` |
| OpenCV | **4.6.0** | 视频编解码与图像处理，默认路径 `/usr/lib/x86_64-linux-gnu/cmake/opencv4` |
| `libyolov11_tensorrt.so` | - | 本项目编译生成，需放置于 `lib/` 目录下 |

> 如果 TensorRT 或 OpenCV 安装路径不同，请修改项目根目录 `CMakeLists.txt` 中的 `TENSORRT_DIR` 和 `OpenCV_DIR`。

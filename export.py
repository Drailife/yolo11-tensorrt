from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Export with fixed batch=4 (no dynamic dims for max engine perf)
export_path = model.export(format="onnx", batch=4, nms=True)

print(f"Model exported to {export_path}")
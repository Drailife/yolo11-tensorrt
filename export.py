from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Export the model to ONNX format with built-in NMS (end-to-end)
export_path = model.export(format="onnx", nms=True)

print(f"Model exported to {export_path}")
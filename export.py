from ultralytics import YOLO

# Load the YOLO model
model = YOLO("model/yolov11n_1920_person_ball_backboard_hoop_20260504.pt")

# Export with fixed batch=4 (no dynamic dims for max engine perf)
export_path = model.export(format="onnx", batch=2)

print(f"Model exported to {export_path}")
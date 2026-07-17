from ultralytics import YOLO

# Load the YOLO model
model = YOLO("model/yolov11n_1920_person_ball_backboard_hoop_20260504.pt")

# Export the model to ONNX format
export_path = model.export(format="onnx")

print(f"Model exported to {export_path}")
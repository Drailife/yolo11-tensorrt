import time
import sys
import cv2
from ultralytics import YOLO


def main(video_path: str, model_path: str = "yolo11s.pt"):
    """
    Run YOLO object detection on a video and display results in real-time.
    Measures and prints total runtime at the end.

    Args:
        video_path: Path to the input video file.
        model_path: Path to the YOLO model file (.pt).
    """
    start_time = time.perf_counter()

    # ----- 1. Load model -----
    print(f"[{time.strftime('%H:%M:%S')}] Loading model: {model_path}")
    model = YOLO(model_path)
    model_load_time = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded. ({(model_load_time - start_time) * 1000:.0f}ms)")

    # ----- 2. Open video -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{time.strftime('%H:%M:%S')}] Video: {video_path}")
    print(f"            FPS: {fps:.2f}, Total frames: {total_frames}")

    # ----- 3. Inference loop -----
    frame_count = 0
    inference_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Run YOLO inference
        results = model(frame, verbose=False)

    # ----- 4. Cleanup & stats -----
    cap.release()
    cv2.destroyAllWindows()
    total_time = time.perf_counter() - start_time
    inference_time = time.perf_counter() - inference_start

    print("\n" + "=" * 50)
    print(f"  Total runtime:          {total_time * 1000:.0f}ms")
    print(f"  Model load time:        {(model_load_time - start_time) * 1000:.0f}ms")
    print(f"  Inference time:         {inference_time * 1000:.0f}ms")
    print(f"  Processed frames:       {frame_count}")
    if frame_count > 0:
        print(f"  Average inference FPS:  {frame_count / inference_time:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_video.py <video_path> [model_path]")
        print("Example: python detect_video.py sample.mp4")
        print("         python detect_video.py sample.mp4 yolo11s.pt")
        sys.exit(1)

    video = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "yolo11s.pt"
    main(video, model)

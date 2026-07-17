import time
import sys
import cv2
import argparse
from ultralytics import YOLO


def main(video_path: str, model_path: str = "yolo11s.pt", batch_size: int = 1):
    """
    Run YOLO object detection on a video with batched inference.
    Measures and prints total runtime at the end.

    Args:
        video_path: Path to the input video file.
        model_path: Path to the YOLO model file (.pt or .engine).
        batch_size: Number of frames to batch together per inference call.
    """
    start_time = time.perf_counter()

    # ----- 1. Load model -----
    print(f"[{time.strftime('%H:%M:%S')}] Loading model: {model_path}")
    model = YOLO(model_path)
    model_load_time = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded. ({(model_load_time - start_time) * 1000:.0f}ms)")
    print(f"[{time.strftime('%H:%M:%S')}] Batch size: {batch_size}")

    # ----- 2. Open video -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{time.strftime('%H:%M:%S')}] Video: {video_path}")
    print(f"            FPS: {fps:.2f}, Total frames: {total_frames}")

    # ----- 3. Batch inference loop -----
    frame_count = 0
    batch_count = 0
    inference_start = time.perf_counter()

    while True:
        # Collect a batch of frames
        frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if not frames:
            break

        batch_count += 1
        frame_count += len(frames)

        # Run batch YOLO inference
        # Ultralytics YOLO supports list-of-frames for batched inference
        results = model(frames, verbose=False)

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
    print(f"  Inference batches:      {batch_count}")
    print(f"  Batch size:             {batch_size}")
    if frame_count > 0:
        avg_batch_time = inference_time / batch_count * 1000 if batch_count > 0 else 0
        print(f"  Average inference FPS:  {frame_count / inference_time:.1f}")
        print(f"  Avg time per batch:     {avg_batch_time:.1f}ms")
        print(f"  Avg time per frame:     {inference_time / frame_count * 1000:.2f}ms")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO batched video inference"
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-m", "--model", default="yolo11s.pt",
        help="Path to YOLO model (.pt or .engine), default: yolo11s.pt"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=1,
        help="Batch size for inference (default: 1, i.e. frame-by-frame)"
    )

    args = parser.parse_args()
    main(args.video, args.model, args.batch)

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Path to the YOLO .pt model (default: yolo11n.pt)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for export (default: 4)")
    parser.add_argument("--nms", type=str, default="true",
                        choices=["true", "false"],
                        help="Whether to include NMS in the exported model (default: true)")

    args = parser.parse_args()

    nms_flag = args.nms.lower() == "true"

    # Load the YOLO model
    model = YOLO(args.model)

    # Export with the specified batch size and NMS flag
    export_path = model.export(format="onnx", batch=args.batch, nms=nms_flag)
    print(f"Model exported to {export_path}")

    # Rename: append suffix like _with_nms_batch4 or _without_nms_batch4
    nms_suffix = "with_nms" if nms_flag else "without_nms"
    new_suffix = f"_{nms_suffix}_batch{args.batch}"

    src = Path(export_path)
    new_name = f"{src.stem}{new_suffix}{src.suffix}"
    dst = src.with_name(new_name)

    os.rename(src, dst)
    print(f"Renamed to {dst}")


if __name__ == "__main__":
    main()
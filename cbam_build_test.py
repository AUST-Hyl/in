"""Quick sanity check: build YOLOv8-CBAM model from YAML."""

from utils import register_cbam_to_yolo
from ultralytics import YOLO


def main():
    register_cbam_to_yolo()
    _ = YOLO("models/yolov8_cbam.yaml")
    print("✓ YOLOv8-CBAM YAML build OK")


if __name__ == "__main__":
    main()


"""
绝缘子破损检测服务
封装 YOLOv8 推理，供 Web API 与 detect.py 共用逻辑
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "data" / "insulator.yaml"

CLASS_NAMES_ZH = {
    "insulator": "绝缘子整串",
    "broken_part": "破损局部",
}

# 推理时使用极低阈值，再按类别分别过滤（破损小目标置信度通常很低）
INFERENCE_CONF = 0.005
DEFAULT_CONF_INSULATOR = 0.25
DEFAULT_CONF_BROKEN = 0.02

BOX_COLORS = {
    0: (0, 200, 0),    # insulator - 绿色
    1: (0, 0, 255),    # broken_part - 红色
}

_RUNTIME: dict[str, Any] = {}


def check_runtime_deps() -> list[str]:
    """检查推理所需依赖，返回缺失的 pip 包名。"""
    required = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
    ]
    missing: list[str] = []
    for module, pip_name in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    return missing


def _cv2():
    if "cv2" not in _RUNTIME:
        import cv2

        _RUNTIME["cv2"] = cv2
    return _RUNTIME["cv2"]


def _np():
    if "np" not in _RUNTIME:
        import numpy as np

        _RUNTIME["np"] = np
    return _RUNTIME["np"]


def _torch():
    if "torch" not in _RUNTIME:
        import torch

        _RUNTIME["torch"] = torch
    return _RUNTIME["torch"]


def _yolo():
    if "YOLO" not in _RUNTIME:
        from ultralytics import YOLO

        _RUNTIME["YOLO"] = YOLO
    return _RUNTIME["YOLO"]


def load_class_names() -> dict[int, str]:
    """从数据集配置加载类别名称。"""
    if DATASET_YAML.exists():
        with open(DATASET_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
    return {0: "insulator", 1: "broken_part"}


def find_available_models() -> list[dict[str, str]]:
    """扫描 runs 目录下的 best.pt 权重文件。"""
    models: list[dict[str, str]] = []
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.exists():
        return models

    for weight_path in sorted(runs_dir.rglob("best.pt")):
        rel = weight_path.relative_to(PROJECT_ROOT)
        exp_name = weight_path.parent.parent.name
        models.append(
            {
                "id": str(rel).replace("\\", "/"),
                "name": exp_name,
                "path": str(weight_path),
            }
        )
    return models


def resolve_default_weights() -> str | None:
    """返回默认模型权重路径。"""
    candidates = [
        PROJECT_ROOT / "runs" / "improved_cbam_cbam" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "baseline" / "weights" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    models = find_available_models()
    return models[0]["path"] if models else None


def decode_image(image_bytes: bytes):
    """将上传的字节解码为 OpenCV BGR 图像。"""
    cv2 = _cv2()
    np = _np()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解析图像，请上传有效的 JPG/PNG 文件")
    return image


def encode_image_bgr(image_bgr, fmt: str = ".jpg") -> str:
    """将 BGR 图像编码为 base64 字符串。"""
    cv2 = _cv2()
    ext = fmt if fmt.startswith(".") else f".{fmt}"
    ok, buffer = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ValueError("图像编码失败")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


class InsulatorDetector:
    """YOLO 检测器单例封装。"""

    def __init__(self) -> None:
        self._model = None
        self._weights: str | None = None
        self._class_names = load_class_names()
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is None:
            torch = _torch()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def weights(self) -> str | None:
        return self._weights

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def load(self, weights: str | None = None) -> None:
        """加载或切换模型权重。"""
        missing = check_runtime_deps()
        if missing:
            raise ImportError(
                "缺少依赖: "
                + ", ".join(missing)
                + "\n请运行: pip install "
                + " ".join(missing)
            )

        path = weights or resolve_default_weights()
        if not path:
            raise FileNotFoundError(
                "未找到模型权重文件。请先训练模型，或通过 --weights 指定 .pt 路径。"
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"模型权重不存在: {path}")

        if self._weights != path:
            YOLO = _yolo()
            self._model = YOLO(path)
            self._weights = path

    def ensure_loaded(self):
        if self._model is None:
            self.load()
        return self._model

    def predict_image(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        conf: float = DEFAULT_CONF_INSULATOR,
        conf_broken: float = DEFAULT_CONF_BROKEN,
        iou: float = 0.45,
        img_size: int = 640,
    ) -> dict[str, Any]:
        """对单张图片执行检测，返回结构化结果。"""
        model = self.ensure_loaded()
        image_bgr = decode_image(image_bytes)

        results = model.predict(
            source=image_bgr,
            imgsz=img_size,
            conf=INFERENCE_CONF,
            iou=iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        raw_detections = self._parse_detections(result)
        detections = self._filter_detections(raw_detections, conf, conf_broken)
        annotated = self._draw_detections(image_bgr, detections)
        broken_count = sum(1 for d in detections if d["class_id"] == 1)

        return {
            "filename": filename,
            "width": int(result.orig_shape[1]),
            "height": int(result.orig_shape[0]),
            "detection_count": len(detections),
            "broken_count": broken_count,
            "has_damage": broken_count > 0,
            "detections": detections,
            "image_original": encode_image_bgr(image_bgr),
            "image_result": encode_image_bgr(annotated),
        }

    def predict_batch(
        self,
        files: list[tuple[str, bytes]],
        conf: float = DEFAULT_CONF_INSULATOR,
        conf_broken: float = DEFAULT_CONF_BROKEN,
        iou: float = 0.45,
        img_size: int = 640,
    ) -> dict[str, Any]:
        """批量检测多张图片。"""
        items = []
        total_detections = 0
        damage_count = 0

        for filename, content in files:
            item = self.predict_image(
                content,
                filename=filename,
                conf=conf,
                conf_broken=conf_broken,
                iou=iou,
                img_size=img_size,
            )
            items.append(item)
            total_detections += item["detection_count"]
            if item["has_damage"]:
                damage_count += 1

        return {
            "total_images": len(items),
            "total_detections": total_detections,
            "damage_images": damage_count,
            "results": items,
        }

    def _filter_detections(
        self,
        detections: list[dict[str, Any]],
        conf_insulator: float,
        conf_broken: float,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for det in detections:
            threshold = conf_broken if det["class_id"] == 1 else conf_insulator
            if det["confidence"] >= threshold:
                filtered.append(det)
        return filtered

    def _draw_detections(self, image_bgr, detections: list[dict[str, Any]]):
        cv2 = _cv2()
        img = image_bgr.copy()
        for det in detections:
            bbox = det["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            color = BOX_COLORS.get(det["class_id"], (255, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name_zh']} {det['confidence']:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return img

    def _parse_detections(self, result) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = self._class_names.get(cls_id, str(cls_id))
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "class_name_zh": CLASS_NAMES_ZH.get(class_name, class_name),
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                }
            )
        return detections


detector = InsulatorDetector()

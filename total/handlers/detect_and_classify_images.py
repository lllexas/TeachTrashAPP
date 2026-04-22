"""
图片推理 handler
使用 models_loader 共享的模型实例
"""

from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

import cv2

from handlers.models_loader import (
    safe_yolo_predict, classify_crop, get_risk_level,
    DETECT_CONF, DETECT_IMGSZ, MAX_BOX_AREA_RATIO, BIG_BOX_MIN_CLS_CONF
)


@dataclass
class DetectionItem:
    bbox: List[int]
    det_conf: float
    cls_name: str
    cls_conf: float
    is_big_box: bool
    risk_level: str


def process_one_image(img_path: Path) -> Optional[Dict]:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"读取失败: {img_path}")
        return None

    start_time = time.time()
    h, w = img.shape[:2]
    img_area = w * h

    results = safe_yolo_predict(
        source=str(img_path),
        conf=DETECT_CONF,
        imgsz=DETECT_IMGSZ
    )

    detections = []
    skipped_big_count = 0

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                det_conf = float(box.conf[0].item())

                x1, y1, x2, y2 = map(int, xyxy)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                box_area = (x2 - x1) * (y2 - y1)
                box_area_ratio = box_area / img_area
                is_big_box = box_area_ratio > MAX_BOX_AREA_RATIO

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_name, cls_conf = classify_crop(crop)

                if is_big_box and cls_conf < BIG_BOX_MIN_CLS_CONF:
                    skipped_big_count += 1
                    continue

                risk_level = get_risk_level(det_conf, cls_conf, is_big_box)

                detections.append({
                    "label": cls_name,
                    "confidence": round(cls_conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "det_conf": round(det_conf, 4),
                    "is_big_box": is_big_box,
                    "risk_level": risk_level,
                })

    # 画框到图片上（用于输出）
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 255, 0) if det["risk_level"] == "high" else (
            (0, 165, 255) if det["risk_level"] == "warning" else (0, 255, 255)
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{det['label']} | det:{det['det_conf']:.2f} cls:{det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(y1 - 8, 20)
        cv2.rectangle(img, (x1, y_text - th - 8), (x1 + tw + 6, y_text + 2), color, -1)
        cv2.putText(img, text, (x1 + 3, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    inference_time_ms = int((time.time() - start_time) * 1000)

    return {
        "filename": img_path.name,
        "inference_time_ms": inference_time_ms,
        "detections": detections,
        "detection_count": len(detections),
        "skipped_big_count": skipped_big_count,
        "annotated_image": img,
    }


def run_image(input_path: str, output_dir: str):
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img_path = Path(input_path)
    result = process_one_image(img_path)

    if result is None:
        return None

    # 保存标注后的图片
    save_path = output_path / img_path.name
    cv2.imwrite(str(save_path), result["annotated_image"])
    result["save_path"] = str(save_path)
    del result["annotated_image"]  # 不返回图片数据

    return result

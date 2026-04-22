from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional

import cv2
import torch
import torchvision
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image


# =========================================================
# 1. 路径配置
# =========================================================
DETECT_MODEL_PATH = "models/YOLO/best.pt"
CLS_MODEL_PATH = "models/ResNet/best_cls_resnet18.pth"

# =========================================================
# 2. 参数配置（图片版允许更高分辨率）
# =========================================================
DETECT_CONF = 0.25
DETECT_IMGSZ = 896
CLS_IMGSZ = 224

MAX_BOX_AREA_RATIO = 0.30
BIG_BOX_MIN_CLS_CONF = 0.55

LOW_CONF_DET = 0.35
LOW_CONF_CLS = 0.55
HIGH_RISK_COUNT = 3

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================================================
# 3. 分类输出权重
# =========================================================
CLASS_SCORE_WEIGHTS = {
    "glass": 0.92,
    "metal": 1.00,
    "organic": 1.00,
    "other": 0.5,
    "paper": 1.28,
    "plastic": 1.00,
}

# =========================================================
# 4. CPU-NMS 兼容补丁
# =========================================================
_original_torchvision_nms = torchvision.ops.nms


def cpu_offload_nms(boxes, scores, iou_threshold):
    original_device = boxes.device
    boxes_cpu = boxes.detach().cpu()
    scores_cpu = scores.detach().cpu()
    keep_indices = _original_torchvision_nms(boxes_cpu, scores_cpu, iou_threshold)
    return keep_indices.to(original_device)


torchvision.ops.nms = cpu_offload_nms


# =========================================================
# 5. 数据结构
# =========================================================
@dataclass
class DetectionItem:
    bbox: List[int]
    det_conf: float
    cls_name: str
    cls_conf: float
    is_big_box: bool
    risk_level: str


# =========================================================
# 6. 设备与模型加载
# =========================================================
PREFERRED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURRENT_DEVICE = PREFERRED_DEVICE
GPU_DISABLED = False

print(f"加载检测模型: {DETECT_MODEL_PATH}")
det_model = YOLO(DETECT_MODEL_PATH)

print(f"加载分类模型: {CLS_MODEL_PATH}")
cls_ckpt = torch.load(CLS_MODEL_PATH, map_location="cpu")
class_names = cls_ckpt["class_names"]

cls_model = models.resnet18(weights=None)
in_features = cls_model.fc.in_features
cls_model.fc = torch.nn.Linear(in_features, len(class_names))
cls_model.load_state_dict(cls_ckpt["model_state_dict"])
cls_model = cls_model.to(CURRENT_DEVICE)
cls_model.eval()

print("分类类别:", class_names)
print("当前运行设备:", CURRENT_DEVICE)

CLASS_WEIGHT_TENSOR = torch.tensor(
    [CLASS_SCORE_WEIGHTS[name] for name in class_names],
    dtype=torch.float32,
    device=CURRENT_DEVICE
)

print("分类重标定权重:")
for name, w in zip(class_names, CLASS_WEIGHT_TENSOR.tolist()):
    print(f"  {name}: {w:.2f}")

cls_transform = transforms.Compose([
    transforms.Resize((CLS_IMGSZ, CLS_IMGSZ)),
    transforms.ToTensor(),
])


def get_yolo_device():
    return 0 if CURRENT_DEVICE == "cuda" else "cpu"


def switch_to_cpu_fallback(reason):
    global CURRENT_DEVICE, GPU_DISABLED, cls_model, CLASS_WEIGHT_TENSOR
    if CURRENT_DEVICE == "cpu":
        return

    print(f"[WARN] GPU 推理失败，切换到 CPU。原因: {reason}")
    CURRENT_DEVICE = "cpu"
    GPU_DISABLED = True
    cls_model = cls_model.to("cpu")
    CLASS_WEIGHT_TENSOR = CLASS_WEIGHT_TENSOR.to("cpu")


def safe_yolo_predict(model, source, conf, imgsz):
    global CURRENT_DEVICE

    try:
        return model.predict(
            source=source,
            conf=conf,
            imgsz=imgsz,
            device=get_yolo_device(),
            verbose=False
        )
    except Exception as e:
        if CURRENT_DEVICE == "cuda":
            switch_to_cpu_fallback(e)
            return model.predict(
                source=source,
                conf=conf,
                imgsz=imgsz,
                device="cpu",
                verbose=False
            )
        raise


# =========================================================
# 7. 工具函数
# =========================================================
def list_images(source: str) -> List[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS])
    return []


def classify_crop(crop_bgr):
    global CURRENT_DEVICE

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    try:
        x = cls_transform(pil_img).unsqueeze(0).to(CURRENT_DEVICE)
        with torch.no_grad():
            logits = cls_model(x)
            probs = torch.softmax(logits, dim=1)
            probs = probs * CLASS_WEIGHT_TENSOR
            probs = probs / probs.sum(dim=1, keepdim=True)
            conf, pred = torch.max(probs, dim=1)
    except Exception as e:
        if CURRENT_DEVICE == "cuda":
            switch_to_cpu_fallback(e)
            x = cls_transform(pil_img).unsqueeze(0).to("cpu")
            with torch.no_grad():
                logits = cls_model(x)
                probs = torch.softmax(logits, dim=1)
                probs = probs * CLASS_WEIGHT_TENSOR
                probs = probs / probs.sum(dim=1, keepdim=True)
                conf, pred = torch.max(probs, dim=1)
        else:
            raise

    return class_names[int(pred.item())], float(conf.item())


def get_risk_level(det_conf: float, cls_conf: float, is_big_box: bool) -> str:
    if is_big_box and cls_conf < 0.60:
        return "warning"
    if det_conf < LOW_CONF_DET or cls_conf < LOW_CONF_CLS:
        return "low"
    return "high"


def draw_result(img, item: DetectionItem):
    x1, y1, x2, y2 = item.bbox

    if item.risk_level == "high":
        color = (0, 255, 0)
    elif item.risk_level == "warning":
        color = (0, 165, 255)
    else:
        color = (0, 255, 255)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    text = f"{item.cls_name} | det:{item.det_conf:.2f} cls:{item.cls_conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(y1 - 8, 20)

    cv2.rectangle(img, (x1, y_text - th - 8), (x1 + tw + 6, y_text + 2), color, -1)
    cv2.putText(
        img,
        text,
        (x1 + 3, y_text - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2
    )


def draw_summary(img, class_counter: Counter, skipped_big_count: int):
    total = sum(class_counter.values())
    y = 30

    cv2.putText(
        img,
        f"total:{total} skipped_big:{skipped_big_count}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    y += 30

    for k, v in sorted(class_counter.items()):
        cv2.putText(
            img,
            f"{k}: {v}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 28

    if total >= HIGH_RISK_COUNT:
        cv2.putText(
            img,
            "warning: possible garbage accumulation",
            (20, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


# =========================================================
# 8. 主流程
# =========================================================
def process_one_image(img_path: Path) -> Optional[Dict]:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"读取失败: {img_path}")
        return None

    h, w = img.shape[:2]
    img_area = w * h

    results = safe_yolo_predict(
        model=det_model,
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

                detections.append(
                    DetectionItem(
                        bbox=[x1, y1, x2, y2],
                        det_conf=det_conf,
                        cls_name=cls_name,
                        cls_conf=cls_conf,
                        is_big_box=is_big_box,
                        risk_level=risk_level,
                    )
                )

    for item in detections:
        draw_result(img, item)

    class_counter = Counter([d.cls_name for d in detections])
    draw_summary(img, class_counter, skipped_big_count)

    save_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(save_path), img)

    return {
        "image_name": img_path.name,
        "save_path": str(save_path),
        "kept_count": len(detections),
        "skipped_big_count": skipped_big_count,
        "class_count": dict(class_counter),
    }


def run_image(input_path: str, output_dir: str):
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_path = Path(input_path)
    return process_one_image(img_path)


def main():
    print("请通过 run_image(input_path, output_dir) 调用该模块。")


if __name__ == "__main__":
    main()
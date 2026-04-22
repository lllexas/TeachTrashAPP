"""
集中式模型加载模块
所有 handler 共用一份模型实例，避免重复加载
"""

from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple

import cv2
import torch
import torchvision
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

# =========================================================
# 1. 路径配置
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DETECT_MODEL_PATH = BASE_DIR / "models" / "YOLO" / "best.pt"
CLS_MODEL_PATH = BASE_DIR / "models" / "ResNet" / "best_cls_resnet18.pth"

# =========================================================
# 2. 参数配置
# =========================================================
DETECT_CONF = 0.25
DETECT_IMGSZ = 768
CLS_IMGSZ = 224

MAX_BOX_AREA_RATIO = 0.30
BIG_BOX_MIN_CLS_CONF = 0.55

LOW_CONF_DET = 0.35
LOW_CONF_CLS = 0.55

# =========================================================
# 3. 分类输出重标定
# =========================================================
CLASS_SCORE_WEIGHTS = {
    "glass": 0.92,
    "metal": 1.00,
    "organic": 1.00,
    "other": 0.50,
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
# 5. 设备与模型加载（全局只执行一次）
# =========================================================
PREFERRED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURRENT_DEVICE = PREFERRED_DEVICE
GPU_DISABLED = False

print(f"[models_loader] 加载检测模型: {DETECT_MODEL_PATH}")
det_model = YOLO(str(DETECT_MODEL_PATH))

print(f"[models_loader] 加载分类模型: {CLS_MODEL_PATH}")
cls_ckpt = torch.load(CLS_MODEL_PATH, map_location="cpu")
class_names = cls_ckpt["class_names"]

cls_model = models.resnet18(weights=None)
in_features = cls_model.fc.in_features
cls_model.fc = torch.nn.Linear(in_features, len(class_names))
cls_model.load_state_dict(cls_ckpt["model_state_dict"])
cls_model = cls_model.to(CURRENT_DEVICE)
cls_model.eval()

print("[models_loader] 分类类别:", class_names)
print("[models_loader] 当前运行设备:", CURRENT_DEVICE)

CLASS_WEIGHT_TENSOR = torch.tensor(
    [CLASS_SCORE_WEIGHTS.get(name, 1.0) for name in class_names],
    dtype=torch.float32,
    device=CURRENT_DEVICE
)

cls_transform = transforms.Compose([
    transforms.Resize((CLS_IMGSZ, CLS_IMGSZ)),
    transforms.ToTensor(),
])

# =========================================================
# 6. 工具函数
# =========================================================
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


def safe_yolo_predict(source, conf=None, imgsz=None):
    global CURRENT_DEVICE
    conf = conf if conf is not None else DETECT_CONF
    imgsz = imgsz if imgsz is not None else DETECT_IMGSZ

    try:
        return det_model.predict(
            source=source,
            conf=conf,
            imgsz=imgsz,
            device=get_yolo_device(),
            verbose=False
        )
    except Exception as e:
        if CURRENT_DEVICE == "cuda":
            switch_to_cpu_fallback(e)
            return det_model.predict(
                source=source,
                conf=conf,
                imgsz=imgsz,
                device="cpu",
                verbose=False
            )
        raise


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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def list_images(source: str) -> List[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS])
    return []


def list_videos(source: str) -> List[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXTS])
    return []


# 导出所有需要共享的符号
__all__ = [
    "det_model", "cls_model", "class_names",
    "get_yolo_device", "safe_yolo_predict",
    "classify_crop", "get_risk_level",
    "list_images", "list_videos",
    "DETECT_CONF", "DETECT_IMGSZ", "CLS_IMGSZ",
    "MAX_BOX_AREA_RATIO", "BIG_BOX_MIN_CLS_CONF",
    "CURRENT_DEVICE"
]

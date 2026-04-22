#!/usr/bin/env python3
"""
垃圾识别推理脚本（WPF 调用版）
用法：
    python inference.py "图片路径" --detect-model "best.pt" --cls-model "best_cls.pth"
输出：JSON 格式的推理结果（写到 stdout）
"""

import argparse
import json
import sys
import os
from pathlib import Path

import cv2
import torch
import torchvision
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

# =========================================================
# 全局：CPU-NMS 兼容补丁
# =========================================================
original_torchvision_nms = torchvision.ops.nms


def cpu_offload_nms(boxes, scores, iou_threshold):
    original_device = boxes.device
    boxes_cpu = boxes.detach().cpu()
    scores_cpu = scores.detach().cpu()
    keep_indices = original_torchvision_nms(boxes_cpu, scores_cpu, iou_threshold)
    return keep_indices.to(original_device)


torchvision.ops.nms = cpu_offload_nms

# =========================================================
# 分类预处理（全局）
# =========================================================
_cls_transform = None


def get_cls_transform(imgsz):
    global _cls_transform
    if _cls_transform is None:
        _cls_transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
        ])
    return _cls_transform


# =========================================================
# 加载分类模型
# =========================================================
_cls_model = None
_class_names = None
_device = None


def load_cls_model(cls_model_path, device):
    global _cls_model, _class_names, _device
    if _cls_model is not None:
        return _cls_model, _class_names, _device

    _device = device
    ckpt = torch.load(cls_model_path, map_location=device)
    _class_names = ckpt["class_names"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(_class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    _cls_model = model
    return _cls_model, _class_names, _device


# =========================================================
# 加载检测模型
# =========================================================
_det_model = None


def load_det_model(det_model_path):
    global _det_model
    if _det_model is not None:
        return _det_model
    _det_model = YOLO(det_model_path)
    return _det_model


# =========================================================
# 工具函数
# =========================================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(source):
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS])
    return []


def classify_crop(crop_bgr, cls_transform, cls_model, class_names, device):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    x = cls_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = cls_model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_idx = int(pred.item())
    pred_conf = float(conf.item())
    pred_name = class_names[pred_idx]
    return pred_name, pred_conf


def draw_result(img, box, det_conf, cls_name, cls_conf, is_big_box=False):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 165, 255) if is_big_box else (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    text = f"{cls_name} | det:{det_conf:.2f} cls:{cls_conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(y1 - 8, 20)

    cv2.rectangle(img, (x1, y_text - th - 8), (x1 + tw + 6, y_text + 2), color, -1)
    cv2.putText(img, text, (x1 + 3, y_text - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


# =========================================================
# 主推理函数
# =========================================================
def infer_image(image_path, det_model, cls_model, class_names, device,
                detect_conf, detect_imgsz, max_box_ratio, big_box_min_cls_conf,
                cls_imgsz, output_dir):
    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "imagePath": str(image_path),
            "hasError": True,
            "errorMessage": "读取图片失败",
            "detections": [],
            "keptCount": 0,
            "skippedBigCount": 0,
        }

    # 检测
    results = det_model.predict(
        source=str(image_path),
        conf=detect_conf,
        imgsz=detect_imgsz,
        device=0 if device == "cuda" else "cpu",
        verbose=False
    )

    h, w = img.shape[:2]
    img_area = w * h

    cls_transform = get_cls_transform(cls_imgsz)

    detections = []
    kept_count = 0
    skipped_big_count = 0

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
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

            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            box_area_ratio = box_area / img_area

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cls_name, cls_conf = classify_crop(crop, cls_transform, cls_model, class_names, device)
            is_big_box = box_area_ratio > max_box_ratio

            item = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "boxWidth": box_w, "boxHeight": box_h,
                "boxAreaRatio": round(box_area_ratio, 4),
                "detectionConfidence": round(det_conf, 4),
                "className": cls_name,
                "classConfidence": round(cls_conf, 4),
                "isBigBox": is_big_box,
                "isFiltered": False,
            }

            if is_big_box and cls_conf < big_box_min_cls_conf:
                item["isFiltered"] = True
                skipped_big_count += 1
                continue

            draw_result(img, [x1, y1, x2, y2], det_conf, cls_name, cls_conf, is_big_box=is_big_box)
            detections.append(item)
            kept_count += 1

    # 保存结果图
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / Path(image_path).name
    cv2.imwrite(str(save_path), img)

    return {
        "imagePath": str(image_path),
        "outputImagePath": str(save_path),
        "hasError": False,
        "errorMessage": None,
        "detections": detections,
        "keptCount": kept_count,
        "skippedBigCount": skipped_big_count,
    }


# =========================================================
# 命令行入口
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="垃圾识别推理脚本")
    parser.add_argument("source", help="输入图片路径或文件夹路径")
    parser.add_argument("--detect-model", required=True, help="YOLO 检测模型路径")
    parser.add_argument("--cls-model", required=True, help="ResNet18 分类模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--imgsz", type=int, default=768, help="检测输入尺寸")
    parser.add_argument("--max-ratio", type=float, default=0.30, help="大框面积比例阈值")
    parser.add_argument("--cls-conf", type=float, default=0.55, help="大框最小分类置信度")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="分类输入尺寸")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    parser.add_argument("--device", default=None, help="计算设备 (cuda/cpu)")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    det_model = load_det_model(args.detect_model)
    cls_model, class_names, _ = load_cls_model(args.cls_model, device)

    images = list_images(args.source)
    if not images:
        print(json.dumps({
            "hasError": True,
            "errorMessage": "未找到可处理的图片"
        }, ensure_ascii=False))
        sys.exit(1)

    # 批量处理时，逐张输出 JSONL（每行一个 JSON）
    # 单张时直接输出一个 JSON
    if len(images) == 1:
        result = infer_image(
            images[0], det_model, cls_model, class_names, device,
            args.conf, args.imgsz, args.max_ratio, args.cls_conf,
            args.cls_imgsz, args.output_dir
        )
        print(json.dumps(result, ensure_ascii=False))
    else:
        for img_path in images:
            result = infer_image(
                img_path, det_model, cls_model, class_names, device,
                args.conf, args.imgsz, args.max_ratio, args.cls_conf,
                args.cls_imgsz, args.output_dir
            )
            print(json.dumps(result, ensure_ascii=False))
            sys.stdout.flush()


if __name__ == "__main__":
    main()

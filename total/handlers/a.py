from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict

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
# 2. 参数配置
# =========================================================
DETECT_CONF = 0.25
DETECT_IMGSZ = 768
CLS_IMGSZ = 224

MAX_BOX_AREA_RATIO = 0.30
BIG_BOX_MIN_CLS_CONF = 0.55

IOU_MATCH_THRESHOLD = 0.30
TRACK_MAX_MISSING = 10
TRACK_MIN_CONFIRM_FRAMES = 2

LOG_EVERY_N_FRAMES = 20
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

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
# 4. CPU-NMS 补丁
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
class TrackState:
    track_id: int
    bbox: List[int]
    last_seen_frame: int
    appear_count: int = 1
    label_votes: Counter = field(default_factory=Counter)
    latest_det_conf: float = 0.0
    latest_cls_conf: float = 0.0

    def update(self, bbox, frame_idx, cls_name, det_conf, cls_conf):
        self.bbox = bbox
        self.last_seen_frame = frame_idx
        self.appear_count += 1
        self.label_votes[cls_name] += 1
        self.latest_det_conf = det_conf
        self.latest_cls_conf = cls_conf

    def best_label(self):
        if not self.label_votes:
            return "unknown"
        return self.label_votes.most_common(1)[0][0]


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

CLASS_WEIGHT_TENSOR = torch.tensor(
    [CLASS_SCORE_WEIGHTS[name] for name in class_names],
    dtype=torch.float32,
    device=CURRENT_DEVICE
)

print("分类类别:", class_names)
print("当前运行设备:", CURRENT_DEVICE)
print("分类重标定权重:")
for name, w in zip(class_names, CLASS_WEIGHT_TENSOR.tolist()):
    print(f"  {name}: {w:.2f}")

cls_transform = transforms.Compose([
    transforms.Resize((CLS_IMGSZ, CLS_IMGSZ)),
    transforms.ToTensor(),
])

frame_idx = 0
next_track_id = 1
tracks: Dict[int, TrackState] = {}


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
def list_videos(source: str) -> List[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXTS])
    return []


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


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


def cleanup_tracks():
    global tracks, frame_idx
    to_delete = []
    for track_id, track in tracks.items():
        if frame_idx - track.last_seen_frame > TRACK_MAX_MISSING:
            to_delete.append(track_id)
    for track_id in to_delete:
        del tracks[track_id]


def match_track(bbox):
    global tracks
    best_iou = 0.0
    best_track_id = None

    for track_id, track in tracks.items():
        iou = compute_iou(bbox, track.bbox)
        if iou > best_iou:
            best_iou = iou
            best_track_id = track_id

    if best_iou >= IOU_MATCH_THRESHOLD:
        return best_track_id
    return None


def draw_result(img, bbox, det_conf, cls_name, cls_conf, track_id):
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    text = f"ID:{track_id} {cls_name} | det:{det_conf:.2f} cls:{cls_conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(y1 - 8, 20)

    cv2.rectangle(img, (x1, y_text - th - 8), (x1 + tw + 6, y_text + 2), color, -1)
    cv2.putText(img, text, (x1 + 3, y_text - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_global_stats(img, global_counter):
    x = img.shape[1] - 240
    y = 30

    cv2.putText(img, "GLOBAL:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 28

    for k, v in sorted(global_counter.items()):
        cv2.putText(img, f"{k}: {v}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        y += 24


def process_video(video_path: Path):
    global frame_idx, next_track_id, tracks
    frame_idx = 0
    next_track_id = 1
    tracks = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = OUTPUT_DIR / f"{video_path.stem}_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    global_counter = Counter()

    print(f"\n开始处理视频: {video_path.name}")
    print(f"尺寸: {width}x{height}, fps:{fps:.2f}, 总帧数:{total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        cleanup_tracks()

        img = frame.copy()
        h, w = img.shape[:2]
        img_area = w * h

        results = safe_yolo_predict(
            model=det_model,
            source=img,
            conf=DETECT_CONF,
            imgsz=DETECT_IMGSZ
        )

        frame_counter = Counter()
        skipped_big = 0
        kept = 0

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

                        bbox = [x1, y1, x2, y2]
                    box_area = (x2 - x1) * (y2 - y1)
                    box_area_ratio = box_area / img_area
                    is_big_box = box_area_ratio > MAX_BOX_AREA_RATIO

                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    cls_name, cls_conf = classify_crop(crop)

                    if is_big_box and cls_conf < BIG_BOX_MIN_CLS_CONF:
                        skipped_big += 1
                        continue

                    bbox = [x1, y1, x2, y2]
                    track_id = match_track(bbox)
                    if track_id is None:
                        track_id = next_track_id
                        next_track_id += 1
                        tracks[track_id] = TrackState(
                            track_id=track_id,
                            bbox=bbox,
                            last_seen_frame=frame_idx,
                            appear_count=1,
                            label_votes=Counter([cls_name]),
                            latest_det_conf=det_conf,
                            latest_cls_conf=cls_conf
                        )
                    else:
                        tracks[track_id].update(
                            bbox=bbox,
                            frame_idx=frame_idx,
                            cls_name=cls_name,
                            det_conf=det_conf,
                            cls_conf=cls_conf
                        )

                    final_label = tracks[track_id].best_label()
                    if tracks[track_id].appear_count < TRACK_MIN_CONFIRM_FRAMES:
                        final_label = cls_name

                    draw_result(img, bbox, det_conf, final_label, cls_conf, track_id)

                    frame_counter[final_label] += 1
                    global_counter[final_label] += 1
                    kept += 1

        info_text = f"frame:{frame_idx} kept:{kept} skipped_big:{skipped_big}"
        cv2.putText(img, info_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        y = 60
        for k, v in sorted(frame_counter.items()):
            cv2.putText(img, f"{k}: {v}",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 26

        draw_global_stats(img, global_counter)
        writer.write(img)

        if frame_idx % LOG_EVERY_N_FRAMES == 0:
            print(f"已处理 {frame_idx}/{total_frames} 帧")

    cap.release()
    writer.release()

    print(f"视频处理完成: {out_path}")
    print(f"总类别统计: {dict(global_counter)}")


def run_video(input_path: str, output_dir: str):
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_path = Path(input_path)
    process_video(video_path)

    return {
        "success": True,
        "input_path": str(video_path),
        "output_dir": str(OUTPUT_DIR)
    }


def main():
    print("请通过 run_video(input_path, output_dir) 调用该模块。")


if __name__ == "__main__":
    main()
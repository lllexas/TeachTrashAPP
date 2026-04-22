"""
摄像头推理 handler
使用 models_loader 共享的模型实例
"""

from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import cv2

from handlers.models_loader import (
    safe_yolo_predict, classify_crop, get_risk_level,
    DETECT_CONF, DETECT_IMGSZ, MAX_BOX_AREA_RATIO, BIG_BOX_MIN_CLS_CONF
)


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


# 摄像头专用参数
DETECT_EVERY_N_FRAMES = 2
IOU_MATCH_THRESHOLD = 0.30
TRACK_MAX_MISSING = 6
TRACK_MIN_CONFIRM_FRAMES = 2


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


def cleanup_tracks(tracks, frame_idx):
    to_delete = []
    for track_id, track in tracks.items():
        if frame_idx - track.last_seen_frame > TRACK_MAX_MISSING:
            to_delete.append(track_id)
    for track_id in to_delete:
        del tracks[track_id]


def match_track(bbox, tracks):
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


def draw_items(img, items, show_record_status=True, enable_record=True):
    frame_counter = Counter()

    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        risk_level = item["risk_level"]

        if risk_level == "high":
            color = (0, 255, 0)
        elif risk_level == "warning":
            color = (0, 165, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = (
            f"ID:{item['track_id']} {item['cls_name']} | "
            f"det:{item['det_conf']:.2f} cls:{item['cls_conf']:.2f}"
        )
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

        frame_counter[item["cls_name"]] += 1

    y = 30
    cv2.putText(
        img,
        f"total:{sum(frame_counter.values())}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    y += 30

    for k, v in sorted(frame_counter.items()):
        cv2.putText(
            img,
            f"{k}:{v}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 25

    if show_record_status:
        status = "REC" if enable_record else "LIVE"
        cv2.putText(
            img,
            status,
            (img.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if enable_record else (255, 255, 255),
            2
        )


def run_camera(
    output_video_path: Optional[str] = None,
    camera_id: int = 0,
    enable_record: bool = True,
    show_window: bool = True,
    max_frames: Optional[int] = None
):
    frame_idx = 0
    next_track_id = 1
    tracks = {}
    last_draw_items = []

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return {"success": False, "error": "无法打开摄像头"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    writer = None
    if enable_record and output_video_path:
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("实时识别开始。")
    if show_window:
        print("按 q 退出，按 r 开/关录制。")

    processed_frames = 0
    last_frame_class_count = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            processed_frames += 1
            cleanup_tracks(tracks, frame_idx)

            img = frame.copy()
            run_full_infer = (frame_idx % DETECT_EVERY_N_FRAMES == 0)

            if run_full_infer:
                h, w = img.shape[:2]
                img_area = w * h
                results = safe_yolo_predict(
                    source=img,
                    conf=DETECT_CONF,
                    imgsz=DETECT_IMGSZ
                )

                new_items = []

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
                                continue

                            track_id = match_track(bbox, tracks)
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

                            risk_level = get_risk_level(det_conf, cls_conf, is_big_box)

                            new_items.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "det_conf": det_conf,
                                "cls_name": final_label,
                                "cls_conf": cls_conf,
                                "is_big_box": is_big_box,
                                "risk_level": risk_level,
                            })

                last_draw_items = new_items
                last_frame_class_count = dict(Counter(item["cls_name"] for item in last_draw_items))

            draw_items(img, last_draw_items, show_record_status=True, enable_record=enable_record)

            if writer is not None:
                writer.write(img)

            if show_window:
                cv2.imshow("Trash Vision Camera", img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("r"):
                    enable_record = not enable_record
                    if enable_record and writer is None and output_video_path:
                        output_path = Path(output_video_path)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                        print("开始录制。")
                    elif not enable_record and writer is not None:
                        writer.release()
                        writer = None
                        print("停止录制。")

            if max_frames is not None and processed_frames >= max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    print("实时识别结束。")

    return {
        "success": True,
        "camera_id": camera_id,
        "frames_processed": processed_frames,
        "record_enabled": enable_record,
        "output_video_path": output_video_path,
        "last_frame_class_count": last_frame_class_count,
    }

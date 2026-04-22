from pathlib import Path
import cv2
import torch
import torchvision
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

# =========================================================
# 1. 模型与路径配置
# =========================================================
# 检测模型（YOLO）
DETECT_MODEL_PATH = "/home/commander/trash_test/runs/garbage_detect_v11l_768_e100_fixed/weights/best.pt"

# 分类模型（ResNet18）
CLS_MODEL_PATH = "/home/commander/trash_test/cls_runs/best_cls_resnet18.pth"

# 输入源：可以是单张图片，也可以是一个文件夹
# 例1：单张图片
# SOURCE = "/mnt/c/Users/23192/Desktop/检测集/test1.jpg"

# 例2：整个文件夹（推荐）
SOURCE = "/mnt/c/Users/23192/Desktop/检测集"

# 输出目录
OUTPUT_DIR = Path("/home/commander/trash_test/final_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. 参数配置
# =========================================================
# 检测阈值：按你的要求，先固定 0.25，不改
DETECT_CONF = 0.25

# 检测输入尺寸
DETECT_IMGSZ = 768

# 大框面积比例阈值
# 如果一个框面积超过整张图的 30%，认为它是“大框”
MAX_BOX_AREA_RATIO = 0.30

# 大框最小分类置信度
# 大框只有在分类模型足够自信时才保留
BIG_BOX_MIN_CLS_CONF = 0.55

# 分类模型输入尺寸
CLS_IMGSZ = 224

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 支持的图片格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================================================
# 3. YOLO 的 CPU-NMS 兼容补丁
#    用于避免你之前遇到的 torchvision CUDA NMS 报错
# =========================================================
original_torchvision_nms = torchvision.ops.nms


def cpu_offload_nms(boxes, scores, iou_threshold):
    """
    将 NMS 放到 CPU 执行，避免部分 CUDA 环境下的兼容性问题
    """
    original_device = boxes.device
    boxes_cpu = boxes.detach().cpu()
    scores_cpu = scores.detach().cpu()
    keep_indices = original_torchvision_nms(boxes_cpu, scores_cpu, iou_threshold)
    return keep_indices.to(original_device)


torchvision.ops.nms = cpu_offload_nms

# =========================================================
# 4. 加载检测模型
# =========================================================
print(f"加载检测模型: {DETECT_MODEL_PATH}")
det_model = YOLO(DETECT_MODEL_PATH)

# =========================================================
# 5. 加载分类模型
# =========================================================
print(f"加载分类模型: {CLS_MODEL_PATH}")
cls_ckpt = torch.load(CLS_MODEL_PATH, map_location=DEVICE)

# 训练时保存的类别名
class_names = cls_ckpt["class_names"]

# 构造 ResNet18 分类模型
cls_model = models.resnet18(weights=None)
in_features = cls_model.fc.in_features
cls_model.fc = torch.nn.Linear(in_features, len(class_names))

# 加载权重
cls_model.load_state_dict(cls_ckpt["model_state_dict"])
cls_model = cls_model.to(DEVICE)
cls_model.eval()

print("分类类别:", class_names)
print("运行设备:", DEVICE)

# =========================================================
# 6. 分类预处理
# =========================================================
cls_transform = transforms.Compose([
    transforms.Resize((CLS_IMGSZ, CLS_IMGSZ)),
    transforms.ToTensor(),
])

# =========================================================
# 7. 工具函数
# =========================================================
def list_images(source):
    """
    返回待处理图片列表
    source 可以是单张图片路径，也可以是文件夹路径
    """
    p = Path(source)

    if p.is_file():
        return [p]

    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS])

    return []


def classify_crop(crop_bgr):
    """
    对裁剪出的垃圾区域做分类
    返回：
        cls_name: 分类类别名
        cls_conf: 分类置信度
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    x = cls_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = cls_model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_idx = int(pred.item())
    pred_conf = float(conf.item())
    pred_name = class_names[pred_idx]

    return pred_name, pred_conf


def draw_result(img, box, det_conf, cls_name, cls_conf, is_big_box=False):
    """
    在原图上画框和文字
    """
    x1, y1, x2, y2 = map(int, box)

    # 颜色：大框用橙色，小框用绿色，方便观察效果
    if is_big_box:
        color = (0, 165, 255)   # 橙色
    else:
        color = (0, 255, 0)     # 绿色

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    text = f"{cls_name} | det:{det_conf:.2f} cls:{cls_conf:.2f}"

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


# =========================================================
# 8. 主流程：检测 + 分类 + 大框软过滤
# =========================================================
def main():
    images = list_images(SOURCE)

    if not images:
        print("没有找到可处理的图片，请检查 SOURCE 路径。")
        return

    print(f"共发现 {len(images)} 张图片，开始处理...")

    for img_path in images:
        print(f"\n处理: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            print("读取失败，跳过。")
            continue

        # YOLO 检测
        results = det_model.predict(
            source=str(img_path),
            conf=DETECT_CONF,
            imgsz=DETECT_IMGSZ,
            device=0 if DEVICE == "cuda" else "cpu",
            verbose=False
        )

        if not results:
            print("未返回检测结果。")
            continue

        result = results[0]
        boxes = result.boxes

        # 如果没检到目标，原图也保存一份
        if boxes is None or len(boxes) == 0:
            print("未检测到垃圾目标。")
            save_path = OUTPUT_DIR / img_path.name
            cv2.imwrite(str(save_path), img)
            print(f"结果已保存: {save_path}")
            continue

        h, w = img.shape[:2]
        img_area = w * h

        kept_count = 0
        skipped_big_count = 0

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            det_conf = float(box.conf[0].item())

            x1, y1, x2, y2 = map(int, xyxy)

            # 防越界
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

            # 先分类
            cls_name, cls_conf = classify_crop(crop)

            # 判断是否是大框
            is_big_box = box_area_ratio > MAX_BOX_AREA_RATIO

            # ==========================
            # 大框软过滤
            # - 大框且分类置信度低：跳过
            # - 其他情况：保留
            # ==========================
            if is_big_box and cls_conf < BIG_BOX_MIN_CLS_CONF:
                skipped_big_count += 1
                continue

            draw_result(
                img,
                [x1, y1, x2, y2],
                det_conf,
                cls_name,
                cls_conf,
                is_big_box=is_big_box
            )
            kept_count += 1

        save_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(save_path), img)

        print(f"保留目标数: {kept_count}")
        print(f"被大框软过滤跳过数: {skipped_big_count}")
        print(f"结果已保存: {save_path}")

    print("\n全部处理完成！")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
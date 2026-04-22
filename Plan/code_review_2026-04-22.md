# 后端代码审查报告

> 日期: 2026/04/22
> 审查人: Claude (小喵)
> 被审查代码: `TrashAPP/total/` 目录

---

## 一、总体评价

后端开发在较短时间内完成了一个功能较完整的 FastAPI 服务，支持图片、视频、摄像头三种输入方式，并且实现了目标跟踪、GPU 回退等进阶功能。**代码结构清晰，注释充分，值得肯定。**

但存在 **2 个严重问题** 和 **5 个建议优化项**，需要修复后才能与前端联调。

---

## 二、严重问题 (必须修复)

### 🔴 P0-1: 模型重复加载 (代码级 Bug)

**问题**: `handlers/detect_and_classify_images.py`、`handlers/detect_and_classify_video.py`、`handlers/a.py` 三个文件各自在模块级别加载了 YOLO 和 ResNet 模型。

**后果**: 服务启动时会加载 **3 份** 模型到内存，造成严重内存浪费（YOLO 模型通常 100MB+，ResNet 几十 MB）。

**证据**:
```python
# detect_and_classify_images.py 第 85-97 行
det_model = YOLO(DETECT_MODEL_PATH)
cls_model = models.resnet18(...)
# detect_and_classify_video.py 第 100-112 行 (重复)
det_model = YOLO(DETECT_MODEL_PATH)
cls_model = models.resnet18(...)
# a.py (camera) 第 105-117 行 (再次重复)
det_model = YOLO(DETECT_MODEL_PATH)
cls_model = models.resnet18(...)
```

**修复建议**:
- 把模型加载提取到单独的 `models_loader.py` 中
- 三个 handler 从 `models_loader` import 已加载的模型
- 或者把模型作为参数传入 handler 函数

---

### 🔴 P0-2: API 返回格式与规范不一致 (接口级 Bug)

**问题**: 当前 `/predict/image` 返回的 JSON 结构与 `api_spec.md` 中定义的不一致。

**当前返回** (嵌套结构):
```json
{
  "success": true,
  "mode": "image",
  "filename": "test.jpg",
  "result": {
    "image_name": "test.jpg",
    "save_path": "outputs/images/test.jpg",
    "kept_count": 3,
    "skipped_big_count": 0,
    "class_count": {"plastic": 2, "paper": 1}
  }
}
```

**规范要求** (扁平结构):
```json
{
  "success": true,
  "filename": "test.jpg",
  "inference_time_ms": 1250,
  "detections": [
    {
      "label": "plastic",
      "confidence": 0.92,
      "bbox": [120, 80, 340, 280]
    }
  ],
  "detection_count": 2
}
```

**后果**: 前端无法解析返回结果，检测结果列表为空。

**修复建议**:
- 返回 `detections` 数组，每个元素包含 `label`, `confidence`, `bbox`
- 添加 `inference_time_ms` 字段
- 移除 `result` 嵌套层

---

### 🔴 P0-3: 视频接口是同步阻塞的 (架构级问题)

**问题**: `/predict/video` 调用的 `run_video()` 会逐帧处理整个视频，可能耗时数分钟。

**后果**: HTTP 请求会超时（通常 30~60 秒），前端收不到响应。

**修复建议** (二选一):
1. **简单方案**: 视频处理改为异步任务，立即返回 `task_id`，前端轮询查询进度
2. **当前可行方案**: 暂时只支持图片接口，视频功能延后实现

---

## 三、建议优化 (推荐修复)

### 🟡 P1-1: `a.py` 文件名不规范

**问题**: 文件名 `a.py` 没有任何语义。

**建议**: 改为 `camera.py` 或 `stream_handler.py`

---

### 🟡 P1-2: 临时文件未清理

**问题**: `main.py` 的 `save_upload_file()` 把上传文件保存到 `temp/` 目录，但从不清理。

**后果**: 长时间运行后磁盘空间被占满。

**建议**: 处理完成后删除临时文件，或使用 `tempfile` 模块的临时文件。

```python
import tempfile
import os

# 处理完成后清理
finally:
    if temp_path.exists():
        temp_path.unlink()
```

---

### 🟡 P1-3: 缺少文件大小限制

**问题**: 可以上传任意大小的文件。

**建议**: 在 FastAPI 中限制上传文件大小，如最大 50MB。

```python
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    # 检查文件大小
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large")
```

---

### 🟡 P1-4: 视频 handler 有缩进错误

**问题**: `detect_and_classify_video.py` 第 362 行 `box_area = ...` 缩进不对，可能导致运行时错误。

**证据**:
```python
                    bbox = [x1, y1, x2, y2]
                box_area = (x2 - x1) * (y2 - y1)  # ← 这里缩进不对！
```

---

### 🟡 P1-5: 未使用 YOLO 的 `tojson()`

**问题**: 代码中手动解析 `box.xyxy[0].tolist()`、`box.conf[0].item()`，而不是使用 YOLOv8 自带的 `results[0].tojson()`。

**影响**: 不是 bug，但代码可以更简单。同时缺少 `results[0].plot()` 这种直接生成标注图的方式。

---

## 四、亮点

| 亮点 | 说明 |
|-----|------|
| ✅ CORS 配置正确 | 允许跨域，前端可以调用 |
| ✅ GPU 回退机制 | CUDA 失败时自动切换到 CPU |
| ✅ 目标跟踪 | 视频/摄像头模式下有 IOU 跟踪 |
| ✅ 分类重标定 | 对不同类别设置不同权重 |
| ✅ 健康检查 | `/health` 接口存在 |
| ✅ 风险等级 | high/warning/low 三级标注 |

---

## 五、修复优先级

```
🔴 P0 (阻塞联调)
  ├── P0-1: 模型重复加载
  ├── P0-2: API 返回格式不一致
  └── P0-3: 视频接口同步阻塞

🟡 P1 (推荐修复)
  ├── P1-1: a.py 文件名
  ├── P1-2: 临时文件未清理
  ├── P1-3: 文件大小限制
  ├── P1-4: 视频缩进错误
  └── P1-5: 未使用 tojson()
```

---

## 六、给后端开发的反馈话术

主人可以直接复制以下内容发给后端开发：

```
代码整体写得不错，结构清晰！但有 3 个必须修的问题，否则前端调不通：

1. 【模型重复加载】三个 handler 文件各自加载了一次模型，内存会爆。
   建议：把模型加载抽到一个单独的 models_loader.py 里。

2. 【返回格式不对】/predict/image 返回的 JSON 结构和我们约定的 API 规范不一致。
   规范里要求返回 detections[] 数组（每个元素有 label/confidence/bbox），
   你现在返回的是嵌套的 result 对象。前端解析不了。

3. 【视频接口阻塞】/predict/video 处理完整个视频才返回，HTTP 会超时。
   建议：先只做图片接口，视频和摄像头延后。

另外还有一个缩进错误在 detect_and_classify_video.py 第 362 行，
box_area 那行缩进不对，运行时可能会报错。
```

# 垃圾识别系统 —— API 规范

> 版本: v1.0
> 协议: HTTP/1.1
> 数据格式: JSON
> 文件上传: multipart/form-data

---

## 一、基础信息

| 项目 | 值 |
|-----|-----|
| 基础 URL | `http://{host}:14785` |
| 文档地址 | `http://{host}:14785/docs` (Swagger UI) |
| 数据格式 | JSON |
| 编码 | UTF-8 |

---

## 二、接口列表

### 2.1 健康检查

检测后端服务是否正常运行。

```
GET /health
```

**请求参数**: 无

**响应示例**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**状态码**:
- `200 OK` - 服务正常
- `503 Service Unavailable` - 服务异常

---

### 2.2 图片推理（核心接口）

接收图片文件，返回 YOLO 检测结果。

```
POST /predict
```

**请求头**:
```
Content-Type: multipart/form-data
```

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|-----|------|
| `file` | File | 是 | 图片文件（jpg, png, bmp, webp）|

**响应示例** (成功):

```json
{
  "success": true,
  "filename": "test.jpg",
  "inference_time_ms": 1250,
  "detections": [
    {
      "label": "recyclable",
      "confidence": 0.9234,
      "bbox": [120, 80, 340, 280]
    },
    {
      "label": "kitchen_waste",
      "confidence": 0.8765,
      "bbox": [400, 150, 580, 400]
    }
  ],
  "detection_count": 2
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|-----|------|------|
| `success` | bool | 是否成功 |
| `filename` | string | 原始文件名 |
| `inference_time_ms` | int | 推理耗时（毫秒）|
| `detections` | array | 检测结果数组 |
| `detections[].label` | string | 垃圾类别标签 |
| `detections[].confidence` | float | 置信度（0~1）|
| `detections[].bbox` | array[int] | 边界框坐标 [x1, y1, x2, y2] |
| `detection_count` | int | 检测到的目标数量 |

**响应示例** (无目标):

```json
{
  "success": true,
  "filename": "empty.jpg",
  "inference_time_ms": 800,
  "detections": [],
  "detection_count": 0
}
```

**响应示例** (失败):

```json
{
  "success": false,
  "error": "Invalid image format",
  "detail": "The uploaded file is not a valid image."
}
```

**状态码**:
- `200 OK` - 推理成功
- `400 Bad Request` - 请求参数错误（如未上传文件）
- `415 Unsupported Media Type` - 文件格式不支持
- `500 Internal Server Error` - 推理过程出错

---

### 2.3 批量推理（可选扩展）

接收多张图片，批量推理。

```
POST /predict/batch
```

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|-----|------|
| `files` | File[] | 是 | 多个图片文件 |

**响应示例**:

```json
{
  "success": true,
  "total": 3,
  "results": [
    {
      "filename": "img1.jpg",
      "detection_count": 2,
      "detections": [...]
    },
    {
      "filename": "img2.jpg",
      "detection_count": 0,
      "detections": []
    }
  ]
}
```

> **注意**: 批量推理可能导致响应时间很长，建议前端使用单图接口逐个调用，配合进度条展示。

---

## 三、错误码规范

| HTTP 状态码 | 错误信息 | 说明 |
|-----------|---------|------|
| 200 | - | 成功 |
| 400 | "No file uploaded" | 请求中未包含文件 |
| 400 | "Empty filename" | 文件名为空 |
| 415 | "Unsupported file type" | 不支持的图片格式 |
| 500 | "Model inference failed" | 模型推理异常 |
| 500 | "Internal server error" | 服务器内部错误 |

---

## 四、CORS 配置要求

后端必须配置 CORS，允许 WPF 客户端跨域访问：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许所有来源（开发阶段）
    allow_methods=["*"],        # 允许所有方法
    allow_headers=["*"],        # 允许所有头
    allow_credentials=False,
)
```

---

## 五、模型加载要求

### 5.1 启动时加载

```python
from ultralytics import YOLO

# ✅ 在应用启动时加载（全局只执行一次）
model = YOLO("models/best.pt")

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 直接使用已加载的模型
    results = model(image)
    ...
```

### 5.2 设备自动选择

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("models/best.pt").to(device)
```

---

## 六、性能要求

| 指标 | 目标值 |
|-----|-------|
| 单图推理时间 (GPU) | ≤ 2 秒 |
| 单图推理时间 (CPU) | ≤ 10 秒 |
| 并发请求 | 支持至少 2 个并发 |
| 内存占用 | ≤ 4GB |

---

## 七、变更记录

| 版本 | 日期 | 变更内容 |
|-----|------|---------|
| v1.0 | 2026/04/22 | 初始版本，定义 /predict 接口 |

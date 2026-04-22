# 垃圾识别系统 —— 团队协作工作指南

> 版本: v1.0
> 日期: 2026/04/22
> 适用: 前端(WPF) + 后端(Python) 协作开发

---

## 一、架构设计

本项目采用 **前后端分离** 架构：

```
┌─────────────────┐      HTTP POST      ┌─────────────────┐
│   WPF 客户端     │  ───────────────▶   │  Python 后端服务 │
│  (C# .NET 8)    │  ◀───────────────   │  (FastAPI)      │
└─────────────────┘    JSON 响应         └─────────────────┘
        │                                        │
        │  - 上传图片                             │  - 加载 YOLO 模型
        │  - 显示结果                             │  - 运行推理
        │  - 用户交互                             │  - 返回 JSON
```

**为什么前后端分离？**
- 模型加载耗时（秒级），每次请求重新加载不可接受
- Python 生态（YOLO、OpenCV）与 C# 直接集成复杂
- 学弟可独立调试后端，无需关心前端 UI
- 未来可扩展为 Web 应用或移动端

---

## 二、角色分工

### 前端负责人（主人）
- WPF 界面开发（输入选择、结果显示、进度反馈）
- HTTP 客户端调用（上传图片、接收 JSON、错误处理）
- 用户体验优化（界面响应、异常提示）

### 后端负责人（学弟）
- FastAPI 服务搭建
- YOLO 模型封装（加载一次，复用推理）
- 接口实现（接收图片 → 推理 → 返回 JSON）

---

## 三、开发流程

### Step 1: 后端先跑通
学弟按本指南的 **API 规范** 实现 FastAPI 服务，确认能独立运行并通过自测。

### Step 2: 提供访问地址
学弟把后端服务的访问地址发给主人，格式如：
```
http://192.168.x.x:8000
```
> **重要**：如果学弟在 WSL (Windows Subsystem for Linux) 中运行后端，IP 地址可能不是 `127.0.0.1`，需要确认实际网络地址。

### Step 3: 前端联调
主人用提供的地址，在 WPF 中调用后端接口，验证图片上传和结果解析。

### Step 4: 集成测试
双方一起测试完整的图片识别流程，处理边界情况（大图片、无目标、网络超时等）。

---

## 四、给学弟的启动指南

### 4.1 技术选型

| 项目 | 选择 | 原因 |
|-----|------|------|
| Web 框架 | **FastAPI** | 高性能、自动 API 文档、AI 友好 |
| 服务器 | **Uvicorn** | ASGI 服务器，异步支持 |
| 通信协议 | HTTP + JSON | 简单、通用、调试方便 |

### 4.2 核心要求

1. **模型只加载一次**
   ```python
   # 错误做法：每次请求都加载
   @app.post("/predict")
   def predict(file):
       model = YOLO("best.pt")  # ❌ 太慢了！
       ...

   # 正确做法：启动时加载，请求时复用
   model = YOLO("best.pt")  # ✅ 全局只加载一次

   @app.post("/predict")
   def predict(file):
       results = model(file)   # ✅ 直接推理
       ...
   ```

2. **允许跨域访问 (CORS)**
   - WPF 浏览器控件或 HTTP 客户端调用时，需要 CORS 支持
   - 配置为允许所有来源（开发阶段）

3. **使用 YOLO 的 `tojson()` 输出**
   - YOLOv8 自带 `results[0].tojson()` 方法，可直接转成标准 JSON
   - 无需手动解析坐标和标签

### 4.3 给学弟的 Prompt 模板

主人可以直接复制以下内容发给学弟，让他转发给 AI 助手（ChatGPT / Claude 等）：

---

**【复制这段给 AI】**

```
我有一个训练好的 YOLOv8 模型（best.pt），现在我需要用 FastAPI 把它封装成一个后端 API。

需求如下：
1. 提供一个 POST /predict 接口，接收客户端上传的图片（multipart/form-data）。
2. 模型应该在服务器启动时只加载一次，不要在每次请求时重新加载。
3. 使用 results[0].tojson() 格式化 YOLO 的输出，并返回给客户端。
4. 需要配置好 CORS，允许所有来源访问。
5. 请给出完整的 Python 代码（main.py），并告诉我如何安装必要的依赖（如 fastapi, uvicorn, python-multipart）。
6. 代码中需要包含正确的异常处理（如图片格式错误、推理失败等）。
```

---

### 4.4 自测 checklist

学弟完成后端后，在提交给主人前，请确认以下事项：

- [ ] 运行 `python main.py`（或 `uvicorn main:app --reload`）能正常启动
- [ ] 浏览器访问 `http://127.0.0.1:8000/docs` 能看到自动生成的 API 文档页面
- [ ] 在 `/docs` 页面中用 "Try it out" 功能上传一张测试图片，能返回 JSON 结果
- [ ] JSON 结果中包含 `boxes`, `labels`, `confidences` 等关键字段
- [ ] 连续上传 10 张图片，每次响应时间在 2 秒以内（GPU）或 10 秒以内（CPU）
- [ ] 终端中没有报错或异常信息
- [ ] 确认后端服务的实际访问地址（如果是 WSL，确认网络 IP）

---

## 五、给主人的前端指南

### 5.1 HTTP 客户端选型

C# 中推荐以下方式调用后端：

| 方式 | 适用场景 | 优点 |
|-----|---------|------|
| `HttpClient` | 简单调用 | .NET 内置，无需额外依赖 |
| `RestSharp` | 复杂场景 | 链式 API，更友好 |

### 5.2 调用示例

```csharp
using var client = new HttpClient();
using var content = new MultipartFormDataContent();

// 添加图片
var fileContent = new StreamContent(File.OpenRead("test.jpg"));
content.Add(fileContent, "file", "test.jpg");

// POST 请求
var response = await client.PostAsync("http://127.0.0.1:8000/predict", content);
var json = await response.Content.ReadAsStringAsync();

// 解析 JSON
var result = JsonSerializer.Deserialize<DetectionResult>(json);
```

### 5.3 前端需要考虑的问题

- **超时处理**：大图片推理慢，设置合理的超时时间（如 30 秒）
- **进度反馈**：批量处理时显示进度条
- **错误处理**：后端未启动、网络不通、图片格式错误等情况的友好提示
- **异步不阻塞 UI**：使用 `async/await`，避免界面卡死

---

## 六、沟通规范

### 6.1 提交物

| 角色 | 提交物 | 格式 |
|-----|-------|------|
| 学弟 | 后端代码 | `main.py` + `requirements.txt` |
| 学弟 | 自测截图 | API 文档页面 + 测试结果截图 |
| 学弟 | 访问地址 | 完整 URL（含 IP 和端口） |
| 主人 | 前端代码 | WPF 项目（GitHub 仓库） |

### 6.2 问题反馈模板

发现 bug 时，请按以下格式描述，便于快速定位：

```
【问题描述】简短描述现象
【复现步骤】1. ... 2. ... 3. ...
【期望结果】应该发生什么
【实际结果】实际发生了什么
【环境信息】操作系统、Python 版本、是否有 GPU
【截图/日志】错误截图或终端输出
```

---

## 七、附录

### A. 推荐的目录结构

```
trash-backend/          ← 学弟的项目
├── main.py             ← FastAPI 入口
├── requirements.txt    ← 依赖列表
├── models/             ← 模型文件夹
│   └── best.pt         ← YOLO 模型
└── test_images/        ← 测试图片（可选）

trash-frontend/         ← 主人的项目 (本仓库)
├── TrashAPP.sln
├── MainWindow.xaml
├── ...
└── Plan/               ← 本文档
```

### B. 依赖安装命令

```bash
pip install fastapi uvicorn python-multipart
# 如果有 YOLO 相关依赖
pip install ultralytics opencv-python
```

### C. 启动命令

```bash
# 开发模式（热重载）
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8000
```

> `--host 0.0.0.0` 很重要！否则 WSL 外无法访问。

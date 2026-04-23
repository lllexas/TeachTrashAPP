# 垃圾识别系统 (TrashAPP)

WPF 前端 + Python FastAPI 后端的垃圾识别展示程序，基于 YOLOv8 检测 + ResNet18 分类。

---

## 目录结构

```
TrashAPP/
├── TrashAPP.sln              # Visual Studio 解决方案
├── MainWindow.xaml           # WPF 主界面
├── ViewModels/               # MVVM 视图模型
├── Core/                     # 核心逻辑（模型、服务）
│   ├── Models/
│   └── Services/
├── total/                    # Python 后端
│   ├── main.py               # FastAPI 入口
│   ├── requirements.txt      # Python 依赖
│   ├── handlers/             # 推理处理模块
│   └── models/               # 模型文件夹（需自行放入）
│       ├── YOLO/
│       └── ResNet/
└── Plan/                     # 项目文档
```

---

## 一、模型文件准备

**需要手动放入以下模型文件：**

| 路径 | 文件 | 说明 |
|------|------|------|
| `total/models/YOLO/` | `best.pt` | YOLOv8 检测模型 |
| `total/models/ResNet/` | `best_cls_resnet18.pth` | ResNet18 分类模型 |

> 模型文件**不要提交到 Git**，已配置 `.gitignore` 忽略。

---

## 二、前端运行（WPF）

1. 用 **Visual Studio 2022** 打开 `TrashAPP.sln`
2. 直接按 **F5** 运行
3. 界面左侧可配置后端地址、输入方式、输出配置
4. 点击**"启动后端"**按钮可一键启动 Python 服务

---

## 三、后端环境搭建

### 3.1 创建虚拟环境

进入后端目录，创建 venv：

```bash
cd total
python -m venv venv
```

### 3.2 安装依赖

```bash
venv\Scripts\pip install -r requirements.txt
```

> 首次安装较慢（torch + ultralytics 约 200MB+），请耐心等待。

### 3.3 手动启动后端

```bash
venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 14785 --no-access-log
```

启动后访问 http://127.0.0.1:14785/docs 查看 API 文档。

---

## 四、端口说明

后端默认端口：**14785**

如需修改，请同时更改以下文件中的端口号：
- `Core/Services/BackendLauncher.cs`
- `Core/Services/HttpInferenceService.cs`
- `ViewModels/MainViewModel.cs`

---

## 五、常见问题

**Q: 一键启动后端提示"找不到 Python"？**
> 请确认 `total/venv/` 已创建且依赖已安装。WPF 会优先使用项目内的 venv。

**Q: 后端提示"No module named fastapi"？**
> 依赖未安装，请执行 `venv\Scripts\pip install -r requirements.txt`。

**Q: 模型文件放哪里？**
> `total/models/YOLO/best.pt` 和 `total/models/ResNet/best_cls_resnet18.pth`

---

## 六、技术栈

| 层级 | 技术 |
|------|------|
| 前端 | WPF (.NET 8), CommunityToolkit.Mvvm |
| 后端 | Python, FastAPI, Uvicorn |
| 模型 | YOLOv8 (ultralytics), ResNet18 (torchvision) |
| 通信 | HTTP/REST, multipart/form-data |

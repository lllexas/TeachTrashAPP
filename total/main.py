from pathlib import Path
import shutil
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from handlers.detect_and_classify_images import run_image
from handlers.detect_and_classify_video import run_video
from handlers.camera_handler import run_camera


# =========================================================
# 1. 基础目录
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "outputs"

IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
CAMERA_OUTPUT_DIR = OUTPUT_DIR / "camera"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. FastAPI 初始化
# =========================================================
app = FastAPI(
    title="Trash Vision Backend",
    description="垃圾识别系统后端接口",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# =========================================================
# 3. 工具函数
# =========================================================
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def save_upload_file(upload_file: UploadFile, save_dir: Path) -> Path:
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / upload_file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    # 检查文件大小
    if file_path.stat().st_size > MAX_FILE_SIZE:
        file_path.unlink()
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    return file_path


def ensure_image_file(filename: str):
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported image file type")


def ensure_video_file(filename: str):
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported video file type")


def cleanup_temp(file_path: Path):
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass


# =========================================================
# 4. 根路由 & 健康检查
# =========================================================
@app.get("/")
def root():
    return {
        "message": "Trash Vision Backend is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "service is running"
    }


# =========================================================
# 5. 图片接口
# =========================================================
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ensure_image_file(file.filename)

    temp_path = None
    try:
        temp_path = save_upload_file(file, TEMP_DIR)
        result = run_image(
            input_path=str(temp_path),
            output_dir=str(IMAGE_OUTPUT_DIR)
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Image processing failed")

        # 返回标准 API 格式
        return {
            "success": True,
            "filename": file.filename,
            "inference_time_ms": result.get("inference_time_ms", 0),
            "detections": result.get("detections", []),
            "detection_count": result.get("detection_count", 0),
            "save_path": result.get("save_path", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image inference failed: {e}")
    finally:
        if temp_path:
            cleanup_temp(temp_path)


# =========================================================
# 6. 视频接口
# =========================================================
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ensure_video_file(file.filename)

    temp_path = None
    try:
        temp_path = save_upload_file(file, TEMP_DIR)
        result = run_video(
            input_path=str(temp_path),
            output_dir=str(VIDEO_OUTPUT_DIR)
        )

        return {
            "success": result.get("success", True),
            "filename": file.filename,
            "output_path": result.get("output_path", ""),
            "total_frames": result.get("total_frames", 0),
            "global_counter": result.get("global_counter", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video inference failed: {e}")
    finally:
        if temp_path:
            cleanup_temp(temp_path)


# =========================================================
# 7. 摄像头接口
# =========================================================
@app.post("/predict/camera")
async def predict_camera(
    camera_id: int = Form(0),
    enable_record: bool = Form(True),
    show_window: bool = Form(False),
    max_frames: int = Form(200)
):
    try:
        output_video_path = None
        if enable_record:
            output_video_path = str(CAMERA_OUTPUT_DIR / "camera_output.mp4")

        result = run_camera(
            output_video_path=output_video_path,
            camera_id=camera_id,
            enable_record=enable_record,
            show_window=show_window,
            max_frames=max_frames
        )

        return {
            "success": result.get("success", True),
            "camera_id": camera_id,
            "frames_processed": result.get("frames_processed", 0),
            "output_video_path": result.get("output_video_path"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera inference failed: {e}")

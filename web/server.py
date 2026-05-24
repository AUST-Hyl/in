"""
绝缘子破损检测 Web 服务
启动: python -m web.server
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.detector import check_runtime_deps, detector, find_available_models, resolve_default_weights

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FastAPI(
    title="绝缘子破损检测",
    description="基于 YOLOv8 的输电线路绝缘子破损检测 Web 服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_image(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix or '未知'}，请上传 JPG/PNG 图片",
        )


@app.get("/api/health")
async def health():
    missing = check_runtime_deps()
    return {
        "status": "ok" if not missing else "degraded",
        "device": detector.device if not missing else None,
        "weights_loaded": detector.weights is not None,
        "weights": detector.weights,
        "missing_deps": missing,
    }


@app.get("/api/models")
async def list_models():
    models = find_available_models()
    default = resolve_default_weights()
    return {"models": models, "default": default}


@app.post("/api/detect")
async def detect_single(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    conf_broken: float = Form(0.02),
    iou: float = Form(0.45),
    img_size: int = Form(640),
    weights: str = Form(""),
):
    """单张图片检测。"""
    _validate_image(file)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    try:
        if weights:
            detector.load(weights)
        result = detector.predict_image(
            content,
            filename=file.filename or "image.jpg",
            conf=conf,
            conf_broken=conf_broken,
            iou=iou,
            img_size=img_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"检测失败: {exc}") from exc

    return {
        "success": True,
        "weights": detector.weights,
        "device": detector.device,
        **result,
    }


@app.post("/api/detect/batch")
async def detect_batch(
    files: list[UploadFile] = File(...),
    conf: float = Form(0.25),
    conf_broken: float = Form(0.02),
    iou: float = Form(0.45),
    img_size: int = Form(640),
    weights: str = Form(""),
):
    """批量图片检测。"""
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一张图片")

    payload: list[tuple[str, bytes]] = []
    for file in files:
        _validate_image(file)
        content = await file.read()
        if content:
            payload.append((file.filename or "image.jpg", content))

    if not payload:
        raise HTTPException(status_code=400, detail="所有上传文件均为空")

    try:
        if weights:
            detector.load(weights)
        batch_result = detector.predict_batch(
            payload,
            conf=conf,
            conf_broken=conf_broken,
            iou=iou,
            img_size=img_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"批量检测失败: {exc}") from exc

    return {
        "success": True,
        "weights": detector.weights,
        "device": detector.device,
        **batch_result,
    }


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _open_browser(host: str, port: int) -> None:
    url_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    time.sleep(1.2)
    webbrowser.open(f"http://{url_host}:{port}")


def main():
    parser = argparse.ArgumentParser(description="绝缘子破损检测 Web 服务")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--weights", type=str, default="", help="模型权重路径")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--open", action="store_true", default=True, help="启动后自动打开浏览器")
    parser.add_argument("--no-open", dest="open", action="store_false", help="不自动打开浏览器")
    args = parser.parse_args()

    missing = check_runtime_deps()
    if missing:
        print("=" * 50)
        print("错误: 缺少以下依赖，Web 服务无法启动检测功能:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n请先安装依赖:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 50)
        sys.exit(1)

    if args.weights:
        detector.load(args.weights)
    else:
        try:
            detector.load()
            print(f"已加载模型: {detector.weights}")
        except FileNotFoundError as exc:
            print(f"警告: {exc}")
            print("服务仍可启动，上传检测时会再次尝试加载模型。")

    print(f"设备: {detector.device}")
    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}"
    print(f"访问地址: {url}")

    if args.open and not args.reload:
        threading.Thread(target=_open_browser, args=(args.host, args.port), daemon=True).start()

    uvicorn.run(
        "web.server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

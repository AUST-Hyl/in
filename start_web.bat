@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   绝缘子破损检测 Web 服务
echo ========================================
echo.

python -c "import cv2, ultralytics, fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖，请稍候...
    pip install opencv-python fastapi "uvicorn[standard]" python-multipart pyyaml ultralytics
    if errorlevel 1 (
        echo.
        echo 依赖安装失败，请手动运行:
        echo   pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo 正在启动服务...
echo 浏览器将自动打开 http://127.0.0.1:8000
echo 关闭此窗口即可停止服务
echo.

python -m web.server --host 127.0.0.1 --port 8000
pause

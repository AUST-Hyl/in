# 基于 YOLOv8 的输电线路绝缘子破损检测系统

## 项目概述

本项目基于 YOLOv8 深度学习模型，实现对输电线路绝缘子的自动破损检测。系统包含基准模型（YOLOv8 Baseline）和改进模型（YOLOv8-CBAM），通过引入 CBAM 注意力机制提升小目标和复杂背景下的检测性能。

## 项目结构

```
insulator/
├── data/                      # 数据集目录
│   ├── images/               # 图像文件
│   │   ├── train/           # 训练集
│   │   ├── val/             # 验证集
│   │   └── test/            # 测试集
│   ├── labels/              # 标注文件（YOLO 格式）
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── insulator.yaml       # 数据集配置文件
├── models/                   # 模型定义
│   ├── cbam.py              # CBAM 注意力机制模块
│   └── yolov8_cbam.yaml     # YOLOv8-CBAM 模型配置
├── train.py                 # 训练脚本
├── val.py                   # 验证脚本
├── detect.py                # 检测脚本
├── requirements.txt         # 依赖包列表
└── README.md               # 项目说明文档
```

## 环境配置

### 系统要求

- 操作系统：Windows / Linux
- Python：3.9+
- CUDA（可选，用于 GPU 训练）

### 安装步骤

1. **克隆或下载项目**

```bash
cd insulator
```

2. **创建虚拟环境（推荐）**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **安装 PyTorch（根据 CUDA 版本）**

```bash
# CPU 版本
pip install torch torchvision

# GPU 版本（CUDA 11.8 示例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 数据集准备

### 数据格式

- **图像格式**：.jpg / .png
- **标注格式**：YOLO 格式 .txt（每行：`class_id x_center y_center width height`，归一化坐标）

### 数据集结构

```
data/
├── images/
│   ├── train/        # 训练图像
│   ├── val/          # 验证图像
│   └── test/         # 测试图像
├── labels/
│   ├── train/        # 训练标注（与图像同名 .txt）
│   ├── val/          # 验证标注
│   └── test/         # 测试标注
└── insulator.yaml    # 数据集配置文件
```

### 类别定义（方案二：整串 + 破损局部，电力巡检标准做法）

- `0`: insulator（绝缘子整串，大框）
- `1`: broken_part（破损的那一片，小框）

**标注规则**：所有绝缘子串标为 insulator，破损的那片单独标为 broken_part。详见 [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)

## 快速开始

### 环境检查

运行快速检查脚本，验证项目配置：

```bash
python quick_start.py
```

该脚本会检查：
- Python 版本和依赖包
- 项目目录结构
- 数据集配置
- GPU/CPU 设备信息

### 1. 训练模型

#### 训练 Baseline 模型

```bash
python train.py --model baseline --model_size s --data data/insulator.yaml --epochs 100 --batch 16
```

#### 训练改进模型（YOLOv8-CBAM）

```bash
python train.py --model cbam --model_size s --data data/insulator.yaml --epochs 100 --batch 16
```

#### 使用 SIoU 损失训练（提升 mAP@0.5:0.95 定位精度）

```bash
python train_improved.py --epochs 200 --batch 8 --loss siou
```

SIoU 相比 CIoU 对框的几何关系建模更精细，有利于小目标（破损区域）的定位精度提升。

#### 训练参数说明

- `--model`: 模型类型（`baseline` 或 `cbam`）
- `--model_size`: 模型规模（`n`, `s`, `m`, `l`, `x`）
- `--data`: 数据集配置文件路径
- `--epochs`: 训练轮数（默认 100）
- `--batch`: 批次大小（默认 16）
- `--img_size`: 输入图像尺寸（默认 640）
- `--device`: 训练设备（`cuda` 或 `cpu`，留空自动选择）
- `--optimizer`: 优化器（`SGD`, `Adam`, `AdamW`，默认 `AdamW`）

### 2. 验证模型

#### 验证单个模型

```bash
python val.py --weights runs/baseline/weights/best.pt --data data/insulator.yaml
```

#### 对比多个模型

```bash
python val.py --compare --baseline runs/baseline/weights/best.pt --cbam runs/yolov8_cbam/weights/best.pt --data data/insulator.yaml
```

#### 验证参数说明

- `--weights`: 模型权重文件路径
- `--data`: 数据集配置文件路径
- `--conf`: 置信度阈值（默认 0.25）
- `--iou`: IoU 阈值（默认 0.45）
- `--compare`: 启用对比模式
- `--baseline`: Baseline 模型路径
- `--cbam`: CBAM 模型路径

### 3. 检测图像/视频

#### 检测单张图像

```bash
python detect.py --weights runs/baseline/weights/best.pt --source path/to/image.jpg --save
```

#### 检测图像目录

```bash
python detect.py --weights runs/baseline/weights/best.pt --source path/to/images/ --save
```

#### 检测视频

```bash
python detect.py --weights runs/baseline/weights/best.pt --source path/to/video.mp4 --save
```

#### 检测参数说明

- `--weights`: 模型权重文件路径（必需）
- `--source`: 输入源（图像/视频/目录，必需）
- `--conf`: 置信度阈值（默认 0.25）
- `--iou`: IoU 阈值（默认 0.45）
- `--save`: 保存检测结果
- `--save_txt`: 保存标签文件
- `--show`: 显示检测结果

## 模型说明

### Baseline 模型

- **模型**：YOLOv8（官方版本）
- **特点**：标准 YOLOv8 结构，无修改
- **用途**：作为性能对比的基准

### YOLOv8-CBAM 改进模型

- **改进点**：在 Backbone 和 Neck 中插入 CBAM 注意力机制
- **优势**：增强对关键特征区域的关注，提升小目标检测能力
- **结构**：保持原有检测头不变，仅增强特征提取能力

## 评估指标

系统使用以下指标评估模型性能：

- **mAP@0.5**：IoU 阈值为 0.5 时的平均精度
- **mAP@0.5:0.95**：IoU 阈值从 0.5 到 0.95 的平均精度
- **Precision**：精确率
- **Recall**：召回率

## 实验结果

训练完成后，实验结果保存在 `runs/` 目录下：

```
runs/
├── baseline/              # Baseline 模型结果
│   ├── weights/          # 模型权重
│   │   ├── best.pt      # 最佳模型
│   │   └── last.pt      # 最后一轮模型
│   ├── results.png       # 训练曲线
│   └── ...
└── yolov8_cbam/          # CBAM 模型结果
    ├── weights/
    └── ...
```

## 常见问题

### 1. CUDA 内存不足

- 减小批次大小：`--batch 8` 或 `--batch 4`
- 减小输入尺寸：`--img_size 512`

### 2. 数据集路径错误

- 检查 `data/insulator.yaml` 中的路径配置
- 确保图像和标注文件对应（同名）

### 3. 模型加载失败

- 确保已安装 `ultralytics` 包
- 检查模型权重文件路径是否正确

## 参考文献

- YOLOv8: https://github.com/ultralytics/ultralytics
- CBAM: Convolutional Block Attention Module (ECCV 2018)

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎提出 Issue。

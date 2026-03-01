# 开发文档

## 📋 目录

1. [项目概述](#项目概述)
2. [技术架构](#技术架构)
3. [代码结构](#代码结构)
4. [核心模块](#核心模块)
5. [开发流程](#开发流程)
6. [API 参考](#api-参考)
7. [常见问题](#常见问题)

---

## 项目概述

### 项目简介

基于 YOLOv8 的输电线路绝缘子破损检测系统，支持：
- ✅ YOLOv8 Baseline 模型
- ✅ YOLOv8-CBAM 改进模型（注意力机制）
- ✅ 多种损失函数（CIoU、SIoU、EIoU、WIoU v3）
- ✅ 小数据集优化训练策略

### 技术栈

- **深度学习框架**: PyTorch + Ultralytics YOLOv8
- **编程语言**: Python 3.9+
- **主要依赖**: torch, ultralytics, opencv-python, numpy

---

## 技术架构

### 系统架构图

```
┌─────────────────────────────────────────┐
│         训练脚本 (train_improved.py)      │
│  ┌──────────────┐  ┌──────────────┐     │
│  │  Baseline    │  │  CBAM模型   │     │
│  │  (YOLOv8)    │  │  (自定义)    │     │
│  └──────────────┘  └──────────────┘     │
│           │              │              │
│           └──────┬───────┘              │
│                  │                       │
│          ┌───────▼────────┐             │
│          │  损失函数模块   │             │
│          │ CIoU/SIoU/EIoU │             │
│          │    /WIoU v3    │             │
│          └───────┬────────┘             │
└──────────────────┼──────────────────────┘
                   │
          ┌────────▼────────┐
          │   工具函数模块    │
          │   (utils.py)     │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │   CBAM模块       │
          │  (models/cbam.py)│
          └──────────────────┘
```

### 数据流

```
数据集 → 数据增强 → 模型训练 → 模型验证 → 模型推理
  ↓         ↓         ↓          ↓          ↓
YOLO格式   Mosaic    YOLOv8     mAP指标    检测结果
          Mixup     CBAM       Precision
          CopyPaste 损失函数    Recall
```

---

## 代码结构

### 目录结构

```
insulator/
├── data/                    # 数据集
│   ├── images/             # 图像文件
│   ├── labels/             # 标注文件
│   └── insulator.yaml      # 数据集配置
├── models/                  # 模型定义
│   ├── cbam.py             # CBAM 注意力模块
│   └── yolov8_cbam.yaml    # CBAM 模型配置
├── docs/                    # 文档
│   ├── DEVELOPMENT_GUIDE.md
│   ├── ADVANCED_LOSS_FUNCTIONS.md
│   └── ...
├── train.py                 # 基础训练脚本
├── train_improved.py        # 改进训练脚本（推荐）
├── val.py                   # 验证脚本
├── detect.py                # 检测脚本
├── utils.py                 # 工具函数
├── analyze_dataset.py        # 数据集分析
└── requirements.txt         # 依赖包
```

### 核心文件说明

| 文件 | 功能 | 说明 |
|------|------|------|
| `train_improved.py` | 训练脚本 | 支持多种损失函数和模型结构 |
| `utils.py` | 工具函数 | 损失函数实现、CBAM注册、设备检查 |
| `models/cbam.py` | CBAM模块 | 注意力机制实现 |
| `models/yolov8_cbam.yaml` | CBAM配置 | YOLOv8-CBAM 模型结构定义 |
| `data/insulator.yaml` | 数据集配置 | 数据集路径和类别定义 |

---

## 核心模块

### 1. 训练模块 (`train_improved.py`)

**主要函数**:

```python
def build_model(args):
    """构建模型（Baseline 或 CBAM）"""
    # 根据 arch 参数选择模型结构
    # 根据 loss 参数应用损失函数 patch

def train_improved(args):
    """改进的训练流程"""
    # 构建模型
    # 配置训练参数（针对小数据集优化）
    # 开始训练
```

**关键参数**:
- `--arch`: 模型结构 (`base` 或 `cbam`)
- `--loss`: 损失函数 (`ciou`, `siou`, `eiou`, `wiou`)
- `--model_size`: 模型规模 (`n`, `s`, `m`, `l`, `x`)
- `--epochs`: 训练轮数
- `--batch`: 批次大小

### 2. 工具模块 (`utils.py`)

**损失函数**:
- `bbox_iou_siou()`: SIoU 损失
- `bbox_iou_eiou()`: EIoU 损失
- `bbox_iou_wiou()`: WIoU v3 损失
- `apply_loss_patch()`: 应用损失函数 patch

**CBAM 注册**:
- `register_cbam_to_yolo()`: 将 CBAM 注册到 Ultralytics

**工具函数**:
- `check_dataset_config()`: 检查数据集配置
- `print_device_info()`: 打印设备信息

### 3. CBAM 模块 (`models/cbam.py`)

**类结构**:
```python
ChannelAttention  # 通道注意力
SpatialAttention  # 空间注意力
CBAM              # 组合注意力模块
```

**使用方式**:
```python
from models.cbam import CBAM
cbam = CBAM(in_planes=128, ratio=16, kernel_size=7)
```

---

## 开发流程

### 1. 环境搭建

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据集准备

确保数据集结构：
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── insulator.yaml
```

### 3. 训练模型

```bash
# Baseline 模型（CIoU 损失）
python train_improved.py --epochs 120 --batch 8 --name baseline

# CBAM 模型
python train_improved.py --arch cbam --epochs 120 --batch 8 --name cbam_model

# 使用 WIoU v3 损失
python train_improved.py --loss wiou --epochs 120 --batch 8 --name wiou_model
```

### 4. 验证模型

```bash
python val.py --weights runs/baseline/weights/best.pt --data data/insulator.yaml
```

### 5. 检测推理

```bash
python detect.py --weights runs/baseline/weights/best.pt --source path/to/image.jpg
```

---

## API 参考

### 训练 API

#### `train_improved(args)`

训练模型的主函数。

**参数**:
- `args.arch`: 模型结构 (`'base'` 或 `'cbam'`)
- `args.loss`: 损失函数 (`'ciou'`, `'siou'`, `'eiou'`, `'wiou'`)
- `args.model_size`: 模型规模 (`'n'`, `'s'`, `'m'`, `'l'`, `'x'`)
- `args.epochs`: 训练轮数
- `args.batch`: 批次大小
- `args.img_size`: 图像尺寸
- `args.lr0`: 初始学习率

**返回**: `results` 对象（包含训练指标）

### 损失函数 API

#### `apply_loss_patch(loss_type)`

应用损失函数 patch。

**参数**:
- `loss_type`: `'siou'`, `'eiou'`, 或 `'wiou'`

**示例**:
```python
from utils import apply_loss_patch
apply_loss_patch('wiou')  # 启用 WIoU v3
```

#### `bbox_iou_wiou(box1, box2, xywh=True, eps=1e-7)`

计算 WIoU v3。

**参数**:
- `box1`: 预测框 (N, 4)
- `box2`: 真实框 (N, 4)
- `xywh`: 是否为 xywh 格式
- `eps`: 防止除零的小常数

**返回**: WIoU 值（标量或张量）

### CBAM API

#### `CBAM(in_planes, ratio=16, kernel_size=7)`

创建 CBAM 模块。

**参数**:
- `in_planes`: 输入通道数
- `ratio`: 通道注意力压缩比例（默认 16）
- `kernel_size`: 空间注意力卷积核大小（默认 7）

**示例**:
```python
from models.cbam import CBAM
cbam = CBAM(in_planes=128, ratio=16, kernel_size=7)
output = cbam(input_tensor)
```

---

## 常见问题

### Q1: 如何添加新的损失函数？

1. 在 `utils.py` 中实现 IoU 计算函数：
```python
def bbox_iou_new_loss(box1, box2, xywh=True, eps=1e-7):
    # 实现你的损失函数
    return iou_value
```

2. 在 `apply_loss_patch()` 中添加新选项：
```python
elif loss_type == 'new_loss':
    iou_fn = bbox_iou_new_loss
    loss_name = "New Loss"
```

3. 在 `train_improved.py` 的 `choices` 中添加：
```python
choices=['ciou', 'siou', 'eiou', 'wiou', 'new_loss']
```

### Q2: 如何修改 CBAM 插入位置？

编辑 `models/yolov8_cbam.yaml`，在需要的位置添加：
```yaml
- [-1, 1, CBAM, [128]]  # 在通道数为 128 的位置插入 CBAM
```

### Q3: 训练时出现 CUDA 内存不足？

- 减小批次大小：`--batch 4`
- 减小图像尺寸：`--img_size 512`
- 使用更小的模型：`--model_size n`

### Q4: 如何自定义数据增强？

在 `train_improved.py` 的 `train_params` 中修改：
```python
'hsv_h': 0.02,      # 色调增强
'degrees': 10.0,    # 旋转角度
'mosaic': 1.0,      # Mosaic 增强
# ... 其他参数
```

### Q5: 如何导出模型为 ONNX？

```python
from ultralytics import YOLO
model = YOLO('runs/baseline/weights/best.pt')
model.export(format='onnx')
```

---

## 扩展开发

### 添加新的注意力机制

1. 在 `models/` 目录创建新文件，如 `models/se.py`
2. 实现注意力模块
3. 在 `utils.py` 中注册：
```python
def register_se_to_yolo():
    from models.se import SE
    import ultralytics.nn.tasks as tasks
    tasks.SE = SE
```

### 添加新的评估指标

在 `val.py` 中添加自定义指标计算逻辑。

### 集成其他检测框架

可以基于现有架构，集成其他检测框架（如 YOLOv5、YOLOv9）进行对比实验。

---

## 调试技巧

### 1. 检查模型结构

```python
from ultralytics import YOLO
model = YOLO('models/yolov8_cbam.yaml')
print(model.model)  # 打印模型结构
```

### 2. 验证数据集

```bash
python analyze_dataset.py
```

### 3. 测试损失函数

```python
from utils import bbox_iou_wiou
import torch
box1 = torch.tensor([[10, 10, 20, 20]])
box2 = torch.tensor([[12, 12, 22, 22]])
iou = bbox_iou_wiou(box1, box2, xywh=False)
print(iou)
```

### 4. 检查设备

```python
from utils import print_device_info
print_device_info()
```

---

## 版本历史

- **v1.0**: 基础 YOLOv8 Baseline 实现
- **v1.1**: 添加 CBAM 注意力机制
- **v1.2**: 支持多种损失函数（SIoU、EIoU、WIoU v3）
- **v1.3**: 优化小数据集训练策略

---

## 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [CBAM 论文](https://arxiv.org/abs/1807.06521)
- [WIoU v3 论文](https://arxiv.org/abs/2301.10051)

---

**最后更新**: 2024

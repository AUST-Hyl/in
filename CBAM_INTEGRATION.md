# YOLOv8-CBAM 集成说明

## 概述

本文档说明如何将 CBAM 注意力机制完整集成到 YOLOv8 模型中。由于 YOLOv8 使用 ultralytics 库，集成自定义模块需要一些额外步骤。

## 方法一：修改 ultralytics 源码（推荐用于研究）

### 步骤 1：找到 ultralytics 安装位置

```python
import ultralytics
print(ultralytics.__file__)
```

### 步骤 2：修改模块注册文件

编辑 `ultralytics/nn/modules/__init__.py`，添加：

```python
from models.cbam import CBAM

__all__ = [..., 'CBAM']  # 在现有列表中添加 CBAM
```

### 步骤 3：使用自定义配置文件

使用 `models/yolov8_cbam.yaml` 配置文件进行训练：

```bash
python train.py --model cbam --data data/insulator.yaml
```

## 方法二：使用 PyTorch 直接构建（推荐用于生产）

### 步骤 1：创建自定义模型类

创建一个新的 Python 文件 `models/yolov8_cbam_model.py`：

```python
import torch
import torch.nn as nn
from ultralytics import YOLO
from models.cbam import CBAM

class YOLOv8CBAM(nn.Module):
    def __init__(self, model_size='s', num_classes=2):
        super().__init__()
        # 加载预训练 YOLOv8 模型
        self.yolo = YOLO(f'yolov8{model_size}.pt')
        # 在关键位置插入 CBAM
        # ... 具体实现
```

### 步骤 2：自定义训练循环

修改训练脚本，使用自定义模型进行训练。

## 方法三：后处理增强（简单但效果有限）

在推理时对特征图应用 CBAM，作为对比实验：

```python
from models.cbam import CBAM

# 在检测后处理中应用 CBAM
cbam = CBAM(channels=256)
enhanced_features = cbam(features)
```

## 当前实现状态

当前项目中的实现：

1. ✅ **CBAM 模块已完整实现** (`models/cbam.py`)
2. ✅ **YOLOv8-CBAM 配置文件已创建** (`models/yolov8_cbam.yaml`)
3. ⚠️ **需要手动集成到 ultralytics**（见方法一）

## 快速开始（使用 Baseline）

如果暂时无法集成 CBAM，可以先使用 Baseline 模型进行训练和实验：

```bash
# 训练 Baseline
python train.py --model baseline --model_size s --epochs 100

# 验证模型
python val.py --weights runs/baseline/weights/best.pt

# 检测图像
python detect.py --weights runs/baseline/weights/best.pt --source path/to/image.jpg
```

## 实验对比建议

1. **Baseline 实验**：使用标准 YOLOv8 训练
2. **改进实验**：集成 CBAM 后训练 YOLOv8-CBAM
3. **对比分析**：使用 `val.py --compare` 对比两个模型

## 注意事项

- YOLOv8 的模块系统相对封闭，自定义模块集成需要修改源码
- 建议在虚拟环境中修改，避免影响其他项目
- 可以 fork ultralytics 仓库进行自定义开发
- 对于毕业设计，可以先完成 Baseline 实验，再逐步集成改进模块

## 参考资源

- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [CBAM 论文](https://arxiv.org/abs/1807.06521)

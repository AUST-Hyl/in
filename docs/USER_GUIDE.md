# 使用文档

## 📋 目录

1. [快速开始](#快速开始)
2. [模型检测](#模型检测)
3. [模型对比](#模型对比)
4. [参数说明](#参数说明)
5. [使用示例](#使用示例)
6. [常见问题](#常见问题)

---

## 快速开始

### 前置要求

- ✅ 已完成模型训练（模型权重文件在 `runs/` 目录下）
- ✅ 已安装所有依赖包
- ✅ 准备好待检测的图像或视频

### 图片存放位置

项目中的图片存放在以下目录：

```
data/images/
├── train/    # 训练集图片（290张）
├── val/      # 验证集图片（83张）
└── test/     # 测试集图片（42张）
```

**使用示例**:
- 检测单张图片: `data/images/test/150.jpg`
- 检测整个测试集: `data/images/test`
- 检测验证集: `data/images/val`

### 模型权重位置

训练完成后，模型权重保存在以下位置：

```
runs/
├── baseline/weights/best.pt          # Baseline 模型（CIoU）
├── baseline_ciou/weights/best.pt    # Baseline 模型（CIoU，改进版）
├── improved_cbam_cbam/weights/best.pt  # CBAM 模型
├── improved_siou/weights/best.pt    # SIoU 损失模型
└── improved_wiou_wiou/weights/best.pt  # WIoU v3 损失模型
```

---

## 模型检测

### 1. 检测单张图像

#### 使用 Baseline 模型

```bash
# 检测单张图片（使用测试集中的图片）
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source data/images/test/150.jpg \
    --save

# 或者使用你自己的图片（需要提供完整路径）
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source path/to/your/image.jpg \
    --save
```

#### 使用 CBAM 模型

```bash
# 检测测试集中的图片
python detect.py \
    --weights runs/improved_cbam_cbam/weights/best.pt \
    --source data/images/test/150.jpg \
    --save
```

#### 使用 WIoU v3 模型

```bash
# 检测测试集中的图片
python detect.py \
    --weights runs/improved_wiou_wiou/weights/best.pt \
    --source data/images/test/150.jpg \
    --save
```

### 2. 批量检测图像目录

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source data/images/test \
    --save \
    --save_txt
```

**说明**:
- `--save`: 保存检测结果图像
- `--save_txt`: 同时保存 YOLO 格式的标签文件

### 3. 检测视频

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source path/to/video.mp4 \
    --save \
    --conf 0.3
```

### 4. 实时显示检测结果

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source path/to/image.jpg \
    --show
```

**注意**: `--show` 参数会在窗口中显示检测结果，适合快速预览。

### 5. 调整检测参数

#### 提高检测精度（降低误检）

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source path/to/image.jpg \
    --conf 0.5 \
    --iou 0.5 \
    --save
```

**参数说明**:
- `--conf 0.5`: 提高置信度阈值，只显示高置信度的检测结果
- `--iou 0.5`: 提高 IoU 阈值，减少重叠框

#### 提高召回率（减少漏检）

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source path/to/image.jpg \
    --conf 0.15 \
    --iou 0.4 \
    --save
```

**参数说明**:
- `--conf 0.15`: 降低置信度阈值，显示更多检测结果
- `--iou 0.4`: 降低 IoU 阈值，允许更多重叠框

---

## 模型对比

### 方法 1: 使用验证脚本对比（推荐）

#### 对比两个模型

```bash
python val.py \
    --compare \
    --baseline runs/baseline/weights/best.pt \
    --cbam runs/improved_cbam_cbam/weights/best.pt \
    --data data/insulator.yaml
```

**输出示例**:
```
==================================================
模型性能对比
==================================================

验证模型: YOLOv8 Baseline
验证模型: YOLOv8-CBAM

======================================================================
模型性能对比表
======================================================================
模型                  mAP@0.5      mAP@0.5:0.95    Precision    Recall
----------------------------------------------------------------------
YOLOv8 Baseline       0.9444        0.5025          0.9473       0.9343
YOLOv8-CBAM           0.9500        0.5200          0.9500       0.9400
======================================================================
```

#### 对比多个损失函数模型

如果需要对比多个模型，可以修改 `val.py` 或使用以下方法：

**步骤 1**: 分别验证每个模型

```bash
# 验证 CIoU 模型
python val.py --weights runs/baseline_ciou/weights/best.pt --data data/insulator.yaml

# 验证 SIoU 模型
python val.py --weights runs/improved_siou/weights/best.pt --data data/insulator.yaml

# 验证 WIoU v3 模型
python val.py --weights runs/improved_wiou_wiou/weights/best.pt --data data/insulator.yaml
```

**步骤 2**: 手动对比结果

记录每个模型的指标，制作对比表格。

### 方法 2: 视觉对比（检测结果对比）

#### 使用不同模型检测同一张图像

```bash
# Baseline 模型
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source test_image.jpg \
    --save \
    --name baseline_result

# CBAM 模型
python detect.py \
    --weights runs/improved_cbam_cbam/weights/best.pt \
    --source test_image.jpg \
    --save \
    --name cbam_result

# WIoU v3 模型
python detect.py \
    --weights runs/improved_wiou_wiou/weights/best.pt \
    --source test_image.jpg \
    --save \
    --name wiou_result
```

检测结果会保存在：
```
runs/detect/
├── baseline_result/
├── cbam_result/
└── wiou_result/
```

然后可以手动对比这些文件夹中的检测结果图像。

### 方法 3: 批量对比测试集

#### 使用不同模型检测整个测试集

```bash
# Baseline 模型
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source data/images/test \
    --save \
    --save_txt \
    --name baseline_test

# CBAM 模型
python detect.py \
    --weights runs/improved_cbam_cbam/weights/best.pt \
    --source data/images/test \
    --save \
    --save_txt \
    --name cbam_test

# WIoU v3 模型
python detect.py \
    --weights runs/improved_wiou_wiou/weights/best.pt \
    --source data/images/test \
    --save \
    --save_txt \
    --name wiou_test
```

然后对比各个文件夹中的检测结果。

---

## 参数说明

### 检测参数 (`detect.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | **必需** | 模型权重文件路径（.pt） |
| `--source` | str | **必需** | 输入源（图像/视频/目录） |
| `--img_size` | int | 640 | 输入图像尺寸 |
| `--conf` | float | 0.25 | 置信度阈值（0-1） |
| `--iou` | float | 0.45 | IoU 阈值（0-1） |
| `--device` | str | auto | 设备（cuda/cpu，留空自动选择） |
| `--save` | flag | True | 保存检测结果 |
| `--save_txt` | flag | False | 保存标签文件（YOLO 格式） |
| `--save_conf` | flag | False | 在标签文件中保存置信度 |
| `--show` | flag | False | 显示检测结果 |
| `--project` | str | runs/detect | 项目输出目录 |
| `--name` | str | exp | 实验名称 |
| `--line_width` | int | 2 | 边界框线宽 |

### 验证参数 (`val.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | - | 模型权重文件路径（单模型验证） |
| `--data` | str | data/insulator.yaml | 数据集配置文件路径 |
| `--img_size` | int | 640 | 输入图像尺寸 |
| `--batch` | int | 16 | 批次大小 |
| `--conf` | float | 0.25 | 置信度阈值 |
| `--iou` | float | 0.45 | IoU 阈值 |
| `--compare` | flag | False | 启用对比模式 |
| `--baseline` | str | - | Baseline 模型路径（对比模式） |
| `--cbam` | str | - | CBAM 模型路径（对比模式） |
| `--save_json` | flag | False | 保存 JSON 格式结果 |
| `--save_hybrid` | flag | False | 保存混合标签 |

---

## 使用示例

### 示例 1: 检测单张图像并保存结果

```bash
# 检测测试集中的图片（使用实际存在的文件名）
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source data/images/test/150.jpg \
    --save \
    --conf 0.25 \
    --name test_detection
```

**结果位置**: `runs/detect/test_detection/150.jpg`

**注意**: 
- 确保图片路径正确，可以使用 `data/images/test/` 目录下的任意图片
- 测试集图片文件名示例: `150.jpg`, `151.jpg`, `1075.jpg` 等

### 示例 2: 批量检测并保存标签文件

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source data/images/test \
    --save \
    --save_txt \
    --save_conf \
    --name batch_detection
```

**结果位置**:
- 检测图像: `runs/detect/batch_detection/`
- 标签文件: `runs/detect/batch_detection/labels/`

### 示例 3: 对比 Baseline 和 CBAM 模型

```bash
python val.py \
    --compare \
    --baseline runs/baseline/weights/best.pt \
    --cbam runs/improved_cbam_cbam/weights/best.pt \
    --data data/insulator.yaml \
    --batch 8
```

### 示例 4: 高精度检测（减少误检）

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source data/images/test \
    --conf 0.6 \
    --iou 0.5 \
    --save \
    --name high_precision
```

### 示例 5: 高召回率检测（减少漏检）

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source data/images/test \
    --conf 0.15 \
    --iou 0.4 \
    --save \
    --name high_recall
```

### 示例 6: 检测视频并实时显示

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source path/to/video.mp4 \
    --show \
    --conf 0.3
```

---

## 常见问题

### Q1: 图片存放在哪个目录？

**A**: 项目中的图片存放在 `data/images/` 目录下：
- **测试集**: `data/images/test/` （42张图片，可用于检测）
- **验证集**: `data/images/val/` （83张图片）
- **训练集**: `data/images/train/` （290张图片）

**检测示例**:
```bash
# 检测测试集中的单张图片
python detect.py --weights runs/baseline/weights/best.pt --source data/images/test/150.jpg --save

# 检测整个测试集
python detect.py --weights runs/baseline/weights/best.pt --source data/images/test --save
```

### Q2: 检测结果保存在哪里？

**A**: 检测结果默认保存在 `runs/detect/` 目录下，以实验名称（`--name`）命名的子文件夹中。

```
runs/detect/
└── exp/              # 默认实验名称
    ├── image1.jpg    # 检测结果图像
    ├── image2.jpg
    └── labels/       # 如果使用了 --save_txt
        ├── image1.txt
        └── image2.txt
```

### Q3: 如何选择合适的置信度阈值？

**A**: 
- **高精度场景**（减少误检）: `--conf 0.5` 或更高
- **平衡场景**（默认）: `--conf 0.25`
- **高召回率场景**（减少漏检）: `--conf 0.15` 或更低

建议先用默认值测试，然后根据实际需求调整。

### Q4: 检测速度很慢怎么办？

**A**: 
1. 使用 GPU: 确保 `--device cuda`（如果有 GPU）
2. 减小图像尺寸: `--img_size 512`
3. 减小批次大小（批量检测时）
4. 使用更小的模型（如 `yolov8n.pt`）

### Q5: 如何对比多个模型的检测结果？

**A**: 
1. **方法 1（推荐）**: 使用验证脚本对比指标
   ```bash
   python val.py --compare --baseline ... --cbam ...
   ```

2. **方法 2**: 使用不同模型检测同一批图像，然后手动对比结果文件夹

3. **方法 3**: 分别验证每个模型，记录指标后制作对比表格

### Q6: 检测结果中类别标签是什么？

**A**: 根据数据集配置（`data/insulator.yaml`）：
- `0`: insulator（绝缘子整串）
- `1`: broken_part（破损局部）

### Q7: 如何只检测特定类别？

**A**: 目前脚本不支持类别过滤，但可以通过后处理实现。或者修改 `detect.py` 添加 `--classes` 参数。

### Q8: 检测视频时如何调整帧率？

**A**: 目前脚本不支持帧率控制。如果需要，可以：
1. 使用 FFmpeg 预处理视频
2. 修改 `detect.py` 添加帧率控制逻辑

### Q9: 如何导出检测结果为其他格式？

**A**: 
- **JSON 格式**: 修改代码使用 `results.json()` 方法
- **COCO 格式**: 使用 `--save_json` 参数（验证时）
- **CSV 格式**: 需要自定义代码处理结果

### Q10: 检测时出现 CUDA 内存不足？

**A**: 
1. 减小图像尺寸: `--img_size 512`
2. 使用 CPU: `--device cpu`
3. 减小批次大小（批量检测时）

### Q11: 如何批量对比多个模型？

**A**: 创建一个简单的对比脚本：

```python
# compare_all_models.py
import subprocess

models = {
    'Baseline': 'runs/baseline/weights/best.pt',
    'CBAM': 'runs/improved_cbam_cbam/weights/best.pt',
    'WIoU': 'runs/improved_wiou_wiou/weights/best.pt',
}

for name, weights in models.items():
    print(f"\n验证模型: {name}")
    subprocess.run([
        'python', 'val.py',
        '--weights', weights,
        '--data', 'data/insulator.yaml'
    ])
```

然后运行：
```bash
python compare_all_models.py
```

---

## 最佳实践

### 1. 检测前检查模型

```bash
# 先验证模型性能
python val.py --weights runs/baseline/weights/best.pt --data data/insulator.yaml
```

### 2. 使用合适的置信度阈值

- 先用默认值 `0.25` 测试
- 根据实际需求调整（精度 vs 召回率）

### 3. 保存检测结果和标签

```bash
python detect.py \
    --weights runs/baseline/weights/best.pt \
    --source data/images/test \
    --save \
    --save_txt \
    --save_conf
```

这样可以：
- 查看检测结果图像
- 分析检测框坐标和置信度
- 进行后续处理和分析

### 4. 对比实验

建议对比实验流程：
1. 使用验证脚本对比指标（mAP、Precision、Recall）
2. 使用不同模型检测同一批图像
3. 手动检查检测结果，找出差异
4. 记录最佳模型和参数

---

## 输出结果说明

### 检测结果图像

检测后的图像会包含：
- **边界框**: 标注检测到的目标
- **类别标签**: insulator 或 broken_part
- **置信度**: 检测的置信度分数

### 标签文件（YOLO 格式）

如果使用 `--save_txt`，会生成 `.txt` 文件，格式：
```
class_id x_center y_center width height [confidence]
```

例如：
```
0 0.5 0.5 0.3 0.4 0.95
1 0.7 0.6 0.1 0.15 0.88
```

### 验证结果指标

- **mAP@0.5**: IoU 阈值为 0.5 时的平均精度
- **mAP@0.5:0.95**: IoU 阈值从 0.5 到 0.95 的平均精度
- **Precision**: 精确率（检测到的目标中，正确的比例）
- **Recall**: 召回率（所有真实目标中，被检测到的比例）

---



"""
改进的训练脚本
针对小数据集和性能问题进行了优化
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from utils import check_dataset_config, print_device_info


def train_improved(args):
    """使用改进策略训练模型"""
    print("=" * 60)
    print("开始训练（改进版）")
    print("=" * 60)
    
    # 如果使用 SIoU 损失，先应用 patch
    if args.loss == 'siou':
        from utils import apply_siou_patch
        apply_siou_patch()
    
    # 加载预训练模型
    model = YOLO(f'yolov8{args.model_size}.pt')
    
    # 改进的训练参数
    # 针对小数据集的优化策略：
    # 1. 增加数据增强强度
    # 2. 使用更小的学习率
    # 3. 增加训练轮数
    # 4. 使用类别权重（如果类别不平衡）
    
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': args.batch,
        'device': args.device,
        'project': args.project,
        'name': f"{args.name}_siou" if args.loss == 'siou' else args.name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': args.optimizer,
        'verbose': True,
        'save': True,
        'save_period': 10,
        
        # 改进参数
        'lr0': args.lr0,  # 初始学习率（降低以稳定训练）
        'lrf': args.lrf,  # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,  # 增加 warmup 轮数
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # 数据增强（针对小数据集增强）
        'hsv_h': 0.02,      # 色调增强
        'hsv_s': 0.7,       # 饱和度增强
        'hsv_v': 0.4,       # 明度增强
        'degrees': 10.0,    # 旋转角度（增加）
        'translate': 0.2,   # 平移（增加）
        'scale': 0.9,       # 缩放范围（增加）
        'shear': 5.0,       # 剪切（增加）
        'perspective': 0.0005,  # 透视变换
        'flipud': 0.0,      # 上下翻转
        'fliplr': 0.5,       # 左右翻转
        'mosaic': 1.0,      # Mosaic 增强
        'mixup': 0.1,       # Mixup 增强（小数据集有用）
        'copy_paste': 0.1,  # Copy-paste 增强
        
        # 损失函数权重（如果类别不平衡，可以调整）
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # 其他
        'close_mosaic': 10,  # 最后10个epoch关闭mosaic
        'amp': True,         # 混合精度训练
        'fraction': 1.0,     # 使用全部数据
    }
    
    # 如果指定了类别权重文件，添加类别权重
    if args.class_weights:
        # YOLOv8 不直接支持类别权重，但可以通过调整 cls 损失权重间接影响
        # 这里我们通过调整 cls 权重来实现
        if args.class_weights == 'auto':
            print("自动计算类别权重...")
            # 这里可以添加自动计算类别权重的逻辑
            pass
    
    print("\n训练参数:")
    print(f"  模型: yolov8{args.model_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch}")
    print(f"  初始学习率: {args.lr0}")
    print(f"  图像尺寸: {args.img_size}")
    print(f"  边界框损失: {args.loss.upper()}（SIoU 可提升 mAP@0.5:0.95）")
    print(f"  数据增强: 已增强（适合小数据集）")
    
    # 开始训练
    results = model.train(**train_params)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"模型保存在: {args.project}/{args.name}/weights/best.pt")
    
    # 打印最终指标
    if hasattr(results, 'results_dict'):
        print("\n最终训练指标:")
        for key, value in results.results_dict.items():
            if 'map' in key.lower() or 'precision' in key.lower() or 'recall' in key.lower():
                print(f"  {key}: {value:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 改进训练脚本（针对小数据集优化）')
    
    # 模型选择
    parser.add_argument('--model_size', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='模型规模')
    
    # 数据集配置
    parser.add_argument('--data', type=str, default='data/insulator.yaml',
                        help='数据集配置文件路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,  # 增加默认轮数
                        help='训练轮数（小数据集建议 200+）')
    parser.add_argument('--batch', type=int, default=8,  # 减小批次大小
                        help='批次大小（小数据集建议 4-8）')
    parser.add_argument('--img_size', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='',
                        help='训练设备')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='优化器类型')
    
    # 学习率参数（改进）
    parser.add_argument('--lr0', type=float, default=0.001,  # 降低初始学习率
                        help='初始学习率（小数据集建议 0.001）')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率因子')
    
    # 输出配置
    parser.add_argument('--project', type=str, default='runs',
                        help='项目输出目录')
    parser.add_argument('--name', type=str, default='improved',
                        help='实验名称')
    
    # 类别权重（可选）
    parser.add_argument('--class_weights', type=str, default=None,
                        help='类别权重文件路径或 "auto" 自动计算')
    
    # 损失函数选择
    parser.add_argument('--loss', type=str, default='siou',
                        choices=['ciou', 'siou'],
                        help='边界框损失函数: ciou(默认) 或 siou(提升定位精度)')
    
    args = parser.parse_args()
    
    # 自动选择设备
    if not args.device:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 打印设备信息
    print_device_info()
    
    print(f"\n模型规模: yolov8{args.model_size}")
    print(f"数据集配置: {args.data}")
    
    # 检查数据集配置文件
    if not check_dataset_config(args.data):
        return
    
    # 开始训练
    train_improved(args)


if __name__ == '__main__':
    main()

"""
YOLOv8 模型验证脚本
用于评估训练好的模型性能
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from utils import check_dataset_config, print_device_info


def validate_model(args):
    """验证模型性能"""
    print("=" * 50)
    print(f"开始验证模型: {args.weights}")
    print("=" * 50)
    
    # 加载训练好的模型
    if not Path(args.weights).exists():
        print(f"错误：模型权重文件 {args.weights} 不存在！")
        return
    
    model = YOLO(args.weights)
    
    # 验证参数
    results = model.val(
        data=args.data,           # 数据集配置文件路径
        imgsz=args.img_size,      # 输入图像尺寸
        batch=args.batch,         # 批次大小
        device=args.device,       # 设备
        conf=args.conf,           # 置信度阈值
        iou=args.iou,             # IoU 阈值
        save_json=args.save_json, # 保存 JSON 格式结果
        save_hybrid=args.save_hybrid,  # 保存混合标签
        plots=True,               # 生成评估图表
        verbose=True,             # 详细输出
    )
    
    # 打印关键指标
    print("\n" + "=" * 50)
    print("验证结果摘要")
    print("=" * 50)
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print("=" * 50)
    
    return results


def compare_models(args):
    """对比多个模型的性能"""
    print("=" * 50)
    print("模型性能对比")
    print("=" * 50)
    
    results_dict = {}
    
    for model_name, weights_path in args.models.items():
        if not Path(weights_path).exists():
            print(f"警告：模型 {model_name} 的权重文件 {weights_path} 不存在，跳过")
            continue
        
        print(f"\n验证模型: {model_name}")
        model = YOLO(weights_path)
        
        results = model.val(
            data=args.data,
            imgsz=args.img_size,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            plots=False,
            verbose=False,
        )
        
        results_dict[model_name] = {
            'mAP@0.5': results.box.map50,
            'mAP@0.5:0.95': results.box.map,
            'Precision': results.box.mp,
            'Recall': results.box.mr,
        }
    
    # 打印对比表格
    print("\n" + "=" * 70)
    print("模型性能对比表")
    print("=" * 70)
    print(f"{'模型':<20} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'Precision':<12} {'Recall':<12}")
    print("-" * 70)
    
    for model_name, metrics in results_dict.items():
        print(f"{model_name:<20} {metrics['mAP@0.5']:<12.4f} {metrics['mAP@0.5:0.95']:<15.4f} "
              f"{metrics['Precision']:<12.4f} {metrics['Recall']:<12.4f}")
    
    print("=" * 70)
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 模型验证脚本')
    
    # 模型权重
    parser.add_argument('--weights', type=str, default='',
                        help='模型权重文件路径（.pt）')
    
    # 数据集配置
    parser.add_argument('--data', type=str, default='data/insulator.yaml',
                        help='数据集配置文件路径')
    
    # 验证参数
    parser.add_argument('--img_size', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='',
                        help='验证设备：cuda 或 cpu（留空自动选择）')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU 阈值')
    
    # 输出选项
    parser.add_argument('--save_json', action='store_true',
                        help='保存 JSON 格式结果')
    parser.add_argument('--save_hybrid', action='store_true',
                        help='保存混合标签')
    
    # 对比模式
    parser.add_argument('--compare', action='store_true',
                        help='对比多个模型（需要指定 --baseline 和 --cbam）')
    parser.add_argument('--baseline', type=str, default='',
                        help='Baseline 模型权重路径')
    parser.add_argument('--cbam', type=str, default='',
                        help='CBAM 模型权重路径')
    
    args = parser.parse_args()
    
    # 自动选择设备
    if not args.device:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 打印设备信息
    print_device_info()
    
    # 检查数据集配置文件
    if not check_dataset_config(args.data):
        return
    
    # 执行验证
    if args.compare:
        # 对比模式
        if not args.baseline or not args.cbam:
            print("错误：对比模式需要指定 --baseline 和 --cbam 参数")
            return
        
        args.models = {
            'YOLOv8 Baseline': args.baseline,
            'YOLOv8-CBAM': args.cbam,
        }
        compare_models(args)
    else:
        # 单模型验证
        if not args.weights:
            print("错误：请指定模型权重文件路径（--weights）")
            return
        validate_model(args)


if __name__ == '__main__':
    main()

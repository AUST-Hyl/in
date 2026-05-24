"""
YOLOv8 检测脚本
用于对图像或视频进行绝缘子破损检测
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from utils import print_device_info


def detect_images(args):
    """对图像进行检测"""
    print("=" * 50)
    print(f"开始检测: {args.source}")
    print("=" * 50)
    
    # 加载模型
    if not Path(args.weights).exists():
        print(f"错误：模型权重文件 {args.weights} 不存在！")
        return
    
    model = YOLO(args.weights)
    
    # 检测参数
    results = model.predict(
        source=args.source,       # 输入源（图像/视频/目录）
        imgsz=args.img_size,      # 输入图像尺寸
        conf=args.conf,           # 置信度阈值
        iou=args.iou,             # IoU 阈值
        device=args.device,       # 设备
        save=args.save,           # 保存检测结果
        save_txt=args.save_txt,   # 保存标签文件
        save_conf=args.save_conf, # 保存置信度
        show=args.show,           # 显示结果
        project=args.project,     # 项目目录
        name=args.name,           # 实验名称
        exist_ok=True,            # 允许覆盖
        line_width=args.line_width,  # 边界框线宽
    )
    
    print("\n检测完成！")
    if args.save:
        print(f"结果保存在: {args.project}/{args.name}")
    
    return results


def detect_video(args):
    """对视频进行检测"""
    print("=" * 50)
    print(f"开始检测视频: {args.source}")
    print("=" * 50)
    
    # 加载模型
    if not Path(args.weights).exists():
        print(f"错误：模型权重文件 {args.weights} 不存在！")
        return
    
    model = YOLO(args.weights)
    
    # 检测参数
    results = model.predict(
        source=args.source,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_txt=False,  # 视频通常不保存标签文件
        save_conf=args.save_conf,
        show=args.show,
        project=args.project,
        name=args.name,
        exist_ok=True,
        line_width=args.line_width,
    )
    
    print("\n检测完成！")
    if args.save:
        print(f"结果保存在: {args.project}/{args.name}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 绝缘子破损检测脚本')
    
    # 模型权重
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重文件路径（.pt）')
    
    # 输入源
    parser.add_argument('--source', type=str, required=True,
                        help='输入源：图像文件、视频文件或目录路径')
    
    # 检测参数
    parser.add_argument('--img_size', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU 阈值')
    parser.add_argument('--device', type=str, default='',
                        help='检测设备：cuda 或 cpu（留空自动选择）')
    
    # 输出选项
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存检测结果')
    parser.add_argument('--save_txt', action='store_true',
                        help='保存标签文件（YOLO 格式）')
    parser.add_argument('--save_conf', action='store_true',
                        help='在标签文件中保存置信度')
    parser.add_argument('--show', action='store_true',
                        help='显示检测结果')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='项目输出目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--line_width', type=int, default=2,
                        help='边界框线宽')
    
    args = parser.parse_args()
    
    # 自动选择设备
    if not args.device:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 打印设备信息
    print_device_info()
    
    print(f"\n模型: {args.weights}")
    print(f"输入源: {args.source}")
    
    # 检查输入源是否存在
    if not Path(args.source).exists():
        print(f"错误：输入源 {args.source} 不存在！")
        return
    
    # 根据文件类型选择检测函数
    source_path = Path(args.source)
    if source_path.is_file():
        if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            detect_video(args)
        else:
            detect_images(args)
    else:
        detect_images(args)


if __name__ == '__main__':
    main()

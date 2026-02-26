"""
快速开始脚本
用于快速验证项目配置和运行示例
"""

import sys
from pathlib import Path
from utils import check_dataset_config, print_device_info, get_device_info


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    # 检查 Python 版本
    python_version = sys.version_info
    print(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 9):
        print("⚠️  警告：建议使用 Python 3.9 或更高版本")
    else:
        print("✓ Python 版本符合要求")
    
    # 检查必要的包
    required_packages = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLOv8',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
    }
    
    print("\n检查依赖包:")
    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (未安装)")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n⚠️  缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
    else:
        print("\n✓ 所有依赖包已安装")
    
    # 检查设备信息
    print_device_info()
    
    return len(missing_packages) == 0


def check_project_structure():
    """检查项目结构"""
    print("\n" + "=" * 60)
    print("项目结构检查")
    print("=" * 60)
    
    required_dirs = [
        'data/images/train',
        'data/images/val',
        'data/images/test',
        'data/labels/train',
        'data/labels/val',
        'data/labels/test',
        'models',
    ]
    
    required_files = [
        'data/insulator.yaml',
        'models/cbam.py',
        'models/yolov8_cbam.yaml',
        'train.py',
        'val.py',
        'detect.py',
        'requirements.txt',
        'README.md',
    ]
    
    print("\n检查目录:")
    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (不存在)")
            missing_dirs.append(dir_path)
    
    print("\n检查文件:")
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (不存在)")
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        print("\n⚠️  项目结构不完整")
        if missing_dirs:
            print(f"缺少目录: {', '.join(missing_dirs)}")
        if missing_files:
            print(f"缺少文件: {', '.join(missing_files)}")
        return False
    else:
        print("\n✓ 项目结构完整")
        return True


def check_dataset():
    """检查数据集配置"""
    print("\n" + "=" * 60)
    print("数据集配置检查")
    print("=" * 60)
    
    config_path = 'data/insulator.yaml'
    if check_dataset_config(config_path):
        # 检查数据集目录是否有数据
        data_dirs = [
            'data/images/train',
            'data/images/val',
        ]
        
        has_data = False
        for data_dir in data_dirs:
            dir_path = Path(data_dir)
            if dir_path.exists():
                image_files = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png'))
                if image_files:
                    print(f"  ✓ {data_dir}: 找到 {len(image_files)} 张图像")
                    has_data = True
                else:
                    print(f"  ⚠️  {data_dir}: 目录为空（需要添加训练数据）")
        
        if not has_data:
            print("\n⚠️  数据集目录为空")
            print("请将训练图像和标注文件放入相应的目录中")
        
        return True
    else:
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("YOLOv8 绝缘子破损检测系统 - 快速检查")
    print("=" * 60 + "\n")
    
    env_ok = check_environment()
    structure_ok = check_project_structure()
    dataset_ok = check_dataset()
    
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    if env_ok and structure_ok:
        print("✓ 项目配置正确，可以开始训练！")
        print("\n下一步:")
        print("1. 准备数据集（图像和标注文件）")
        print("2. 运行训练: python train.py --model baseline --model_size s")
        print("3. 验证模型: python val.py --weights runs/baseline/weights/best.pt")
        print("4. 检测图像: python detect.py --weights <模型路径> --source <图像路径>")
    else:
        print("⚠️  项目配置不完整，请先解决上述问题")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

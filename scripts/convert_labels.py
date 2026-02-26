"""
标注格式转换脚本
将旧的 normal_insulator / broken_insulator 格式转换为新的 insulator / broken_part 格式

注意：此脚本只能处理部分转换，broken_insulator 的标注需要手动重新标注！
- 旧 class 0 (normal_insulator) → 新 class 0 (insulator)：可直接转换，框不变
- 旧 class 1 (broken_insulator) → 需要手动操作：
  - 原框可能是整串，需改为 insulator
  - 需要新画 broken_part 小框（仅框破损片）
"""

import argparse
from pathlib import Path
from collections import defaultdict


def analyze_labels(label_dir):
    """分析标注文件中的类别分布"""
    stats = defaultdict(lambda: {'count': 0, 'files': []})
    
    for label_file in Path(label_dir).glob('*.txt'):
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    stats[class_id]['count'] += 1
                    if label_file.name not in stats[class_id]['files']:
                        stats[class_id]['files'].append(label_file.name)
    
    return dict(stats)


def convert_normal_to_insulator(label_path, output_path=None):
    """
    将 normal_insulator (0) 转换为 insulator (0)
    实际上 class 0 不变，只是语义从"正常绝缘子"变为"绝缘子整串"
    """
    if output_path is None:
        output_path = label_path
    
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            # class 0: normal_insulator -> insulator (仍然是 0)
            # class 1: broken_insulator -> 需要手动处理
            if class_id == 0:
                new_lines.append(line)  # 保持不变
            elif class_id == 1:
                # broken_insulator: 将原框改为 insulator（假设原框是整串）
                # 注意：这样会丢失 broken_part！需要手动添加
                new_line = '0 ' + ' '.join(parts[1:5]) + '\n'
                new_lines.append(new_line)
                print(f"  警告: {label_path.name} 中的 broken_insulator 已改为 insulator，"
                      f"需要手动添加 broken_part 小框！")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(description='标注格式转换脚本')
    parser.add_argument('--labels_dir', type=str, default='data/labels',
                        help='标注文件目录（包含 train/val/test 子目录）')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅分析，不实际转换')
    parser.add_argument('--backup', action='store_true', default=True,
                        help='转换前备份原文件')
    
    args = parser.parse_args()
    
    labels_dir = Path(args.labels_dir)
    
    print("=" * 60)
    print("标注文件分析")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
        
        stats = analyze_labels(split_dir)
        print(f"\n{split}/ 目录:")
        for class_id in sorted(stats.keys()):
            old_name = 'normal_insulator' if class_id == 0 else 'broken_insulator'
            new_name = 'insulator' if class_id == 0 else 'broken_part (需手动)'
            print(f"  class {class_id} ({old_name}): {stats[class_id]['count']} 个框, "
                  f"{len(stats[class_id]['files'])} 个文件")
            print(f"    -> 新: {new_name}")
    
    print("\n" + "=" * 60)
    print("转换说明")
    print("=" * 60)
    print("""
类别映射：
  - 旧 class 0 (normal_insulator) -> 新 class 0 (insulator): 可直接转换
  - 旧 class 1 (broken_insulator) -> 需要手动重新标注！
    - 原框改为 insulator（整串）
    - 新画 broken_part 小框（仅框破损片）

建议操作：
  1. 备份 data/labels 目录
  2. 使用 LabelImg 等工具打开有 broken_insulator 的图片
  3. 对每张图：添加 insulator 大框 + broken_part 小框
  4. 删除或修改旧的整串框

数据集配置文件 (data/insulator.yaml) 已更新为新类别名称。
""")


if __name__ == '__main__':
    main()

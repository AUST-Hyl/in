"""
数据集分析脚本
分析数据集规模、类别分布等信息
"""

from pathlib import Path
from collections import Counter


def analyze_dataset():
    """分析数据集"""
    print("=" * 60)
    print("数据集分析")
    print("=" * 60)
    
    # 统计图像数量
    train_dir = Path('data/images/train')
    val_dir = Path('data/images/val')
    test_dir = Path('data/images/test')
    
    train_imgs = len(list(train_dir.glob('*.jpg'))) + len(list(train_dir.glob('*.png')))
    val_imgs = len(list(val_dir.glob('*.jpg'))) + len(list(val_dir.glob('*.png')))
    test_imgs = len(list(test_dir.glob('*.jpg'))) + len(list(test_dir.glob('*.png')))
    
    total_imgs = train_imgs + val_imgs + test_imgs
    
    print(f"\n图像数量统计:")
    print(f"  训练集: {train_imgs} 张")
    print(f"  验证集: {val_imgs} 张")
    print(f"  测试集: {test_imgs} 张")
    print(f"  总计: {total_imgs} 张")
    
    # 统计标注文件
    train_labels = Path('data/labels/train')
    val_labels = Path('data/labels/val')
    test_labels = Path('data/labels/test')
    
    train_label_files = len(list(train_labels.glob('*.txt')))
    val_label_files = len(list(val_labels.glob('*.txt')))
    test_label_files = len(list(test_labels.glob('*.txt')))
    
    print(f"\n标注文件数量:")
    print(f"  训练集: {train_label_files} 个")
    print(f"  验证集: {val_label_files} 个")
    print(f"  测试集: {test_label_files} 个")
    
    # 分析类别分布
    print(f"\n类别分布分析:")
    
    def count_classes(label_dir, split_name):
        class_counter = Counter()
        total_objects = 0
        empty_files = 0
        
        for label_file in label_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        empty_files += 1
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counter[class_id] += 1
                            total_objects += 1
            except Exception as e:
                print(f"  警告：读取 {label_file} 失败: {e}")
        
        return class_counter, total_objects, empty_files
    
    train_classes, train_objects, train_empty = count_classes(train_labels, '训练集')
    val_classes, val_objects, val_empty = count_classes(val_labels, '验证集')
    test_classes, test_objects, test_empty = count_classes(test_labels, '测试集')
    
    print(f"\n训练集:")
    print(f"  总目标数: {train_objects}")
    print(f"  空标注文件: {train_empty}")
    for class_id in sorted(train_classes.keys()):
        count = train_classes[class_id]
        percentage = (count / train_objects * 100) if train_objects > 0 else 0
        class_name = 'insulator' if class_id == 0 else 'broken_part'
        print(f"  类别 {class_id} ({class_name}): {count} 个 ({percentage:.1f}%)")
    
    print(f"\n验证集:")
    print(f"  总目标数: {val_objects}")
    print(f"  空标注文件: {val_empty}")
    for class_id in sorted(val_classes.keys()):
        count = val_classes[class_id]
        percentage = (count / val_objects * 100) if val_objects > 0 else 0
        class_name = 'insulator' if class_id == 0 else 'broken_part'
        print(f"  类别 {class_id} ({class_name}): {count} 个 ({percentage:.1f}%)")
    
    # 问题诊断
    print(f"\n" + "=" * 60)
    print("问题诊断")
    print("=" * 60)
    
    issues = []
    suggestions = []
    
    if total_imgs < 100:
        issues.append(f"数据集太小（仅 {total_imgs} 张图像）")
        suggestions.append("建议：增加数据集到至少 200-500 张图像")
    
    if train_imgs < 50:
        issues.append(f"训练集太小（仅 {train_imgs} 张）")
        suggestions.append("建议：训练集至少需要 100+ 张图像")
    
    if train_objects == 0:
        issues.append("训练集没有标注目标")
        suggestions.append("建议：检查标注文件是否正确")
    else:
        # 检查类别平衡
        if len(train_classes) == 2:
            class_0_count = train_classes.get(0, 0)
            class_1_count = train_classes.get(1, 0)
            ratio = max(class_0_count, class_1_count) / min(class_0_count, class_1_count) if min(class_0_count, class_1_count) > 0 else float('inf')
            if ratio > 5:
                issues.append(f"类别严重不平衡（比例 {ratio:.1f}:1）")
                suggestions.append("建议：使用类别权重或数据增强平衡类别")
    
    if issues:
        print("\n发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n改进建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print("\n✓ 数据集基本正常")
    
    print("=" * 60)
    
    return {
        'train_imgs': train_imgs,
        'val_imgs': val_imgs,
        'test_imgs': test_imgs,
        'train_objects': train_objects,
        'val_objects': val_objects,
        'train_classes': train_classes,
        'val_classes': val_classes,
    }


if __name__ == '__main__':
    analyze_dataset()

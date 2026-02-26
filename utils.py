"""
工具函数
用于辅助 YOLOv8 模型训练和评估
"""

import math
import torch
from pathlib import Path
import yaml


def bbox_iou_siou(box1, box2, xywh=True, eps=1e-7):
    """
    SIoU (Scylla IoU) - 用于替代 CIoU，提升定位精度
    参考论文: SIoU Loss: More Powerful Learning for Bounding Box Regression (2022)
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1 = b1_x2 - b1_x1 + eps
        h1 = b1_y2 - b1_y1 + eps
        w2 = b2_x2 - b2_x1 + eps
        h2 = b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2

    sigma = torch.pow((b2_cy - b1_cy).pow(2) + (b2_cx - b1_cx).pow(2), 0.5)
    sin_alpha = torch.clamp(sigma / (ch + eps), 0, 1)
    angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)

    rho_x = ((b2_cx - b1_cx) / (cw + eps)).pow(2)
    rho_y = ((b2_cy - b1_cy) / (ch + eps)).pow(2)
    gamma = 2 - angle_cost
    distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

    omega_w = torch.abs(w1 - w2) / (torch.maximum(w1, w2) + eps)
    omega_h = torch.abs(h1 - h2) / (torch.maximum(h1, h2) + eps)
    shape_cost = torch.pow(1 - torch.exp(-omega_w), 4) + torch.pow(1 - torch.exp(-omega_h), 4)

    siou = iou - 0.5 * (distance_cost + shape_cost)
    return siou.squeeze(-1) if siou.shape[-1] == 1 else siou


def apply_siou_patch():
    """将 YOLOv8 的 CIoU 替换为 SIoU，在训练前调用"""
    import ultralytics.utils.loss as loss_module
    from ultralytics.utils.tal import bbox2dist
    import torch.nn.functional as F

    def patched_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                       target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou_siou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
        if iou.dim() > 1:
            iou = iou.squeeze(-1)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl

    loss_module.BboxLoss.forward = patched_forward
    print("✓ SIoU 损失函数已启用（已替换 CIoU）")
    return True


def register_cbam_to_yolo():
    """
    将 CBAM 模块注册到 YOLOv8 的模块字典中
    需要在训练前调用
    """
    try:
        from ultralytics.nn.modules import Conv, C2f, SPPF, Concat, Detect
        from ultralytics.utils.torch_utils import fuse_conv_and_bn
        from models.cbam import CBAM
        
        # 尝试注册 CBAM 到 ultralytics 的模块字典
        # 注意：这可能需要根据 ultralytics 版本调整
        import ultralytics.nn.modules as modules
        
        if not hasattr(modules, 'CBAM'):
            modules.CBAM = CBAM
        
        print("CBAM 模块已注册到 YOLOv8")
        return True
    except Exception as e:
        print(f"警告：无法自动注册 CBAM 模块: {e}")
        print("将使用替代方案进行训练")
        return False


def check_dataset_config(config_path):
    """
    检查数据集配置文件是否正确
    
    Args:
        config_path: 数据集配置文件路径
    
    Returns:
        bool: 配置文件是否有效
    """
    if not Path(config_path).exists():
        print(f"错误：数据集配置文件 {config_path} 不存在")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                print(f"错误：数据集配置文件缺少必需的键: {key}")
                return False
        
        print(f"数据集配置验证通过:")
        print(f"  类别数量: {config['nc']}")
        print(f"  类别名称: {config['names']}")
        return True
    except Exception as e:
        print(f"错误：读取数据集配置文件失败: {e}")
        return False


def get_device_info():
    """
    获取设备信息
    
    Returns:
        dict: 设备信息字典
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    if info['cuda_available']:
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
    
    return info


def print_device_info():
    """打印设备信息"""
    info = get_device_info()
    print("=" * 50)
    print("设备信息")
    print("=" * 50)
    print(f"CUDA 可用: {info['cuda_available']}")
    print(f"使用设备: {info['device']}")
    
    if info['cuda_available']:
        print(f"CUDA 版本: {info['cuda_version']}")
        print(f"GPU 名称: {info['gpu_name']}")
        print(f"GPU 数量: {info['gpu_count']}")
    print("=" * 50)

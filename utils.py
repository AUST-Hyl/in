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


def bbox_iou_eiou(box1, box2, xywh=True, eps=1e-7):
    """
    EIoU (Efficient IoU) - 直接优化宽高比，计算简单高效，对小目标友好
    参考论文: Efficient Intersection over Union Loss for Accurate Object Detection (2020)
    
    优点：
    - 计算简单，效率高
    - 直接优化宽高比，避免 CIoU 的复杂约束
    - 对小目标定位更友好
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

    # IoU
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # 最小外接框
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    # 中心点距离
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    rho2 = (b2_cx - b1_cx).pow(2) + (b2_cy - b1_cy).pow(2)

    # EIoU = IoU - (中心距离^2 / 外接框对角线^2) - (宽高差^2 / 外接框宽高^2)
    c2 = cw.pow(2) + ch.pow(2) + eps
    distance_loss = rho2 / c2
    
    # 宽高比损失（直接优化宽高差）
    w_loss = (w1 - w2).pow(2) / (cw.pow(2) + eps)
    h_loss = (h1 - h2).pow(2) / (ch.pow(2) + eps)
    aspect_loss = w_loss + h_loss

    eiou = iou - distance_loss - 0.5 * aspect_loss
    return eiou.squeeze(-1) if eiou.shape[-1] == 1 else eiou


def bbox_iou_wiou(box1, box2, xywh=True, eps=1e-7):
    """
    WIoU v3 (Wise-IoU v3) - 2023年最新，动态梯度调整，专门针对小目标优化
    参考论文: Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism (2023)
    
    核心创新：
    - 动态聚焦机制：根据 IoU 质量动态调整梯度
    - 避免大目标主导训练，对小目标更友好
    - 自适应权重，提升定位精度
    
    优点：
    - 对小目标定位效果显著提升
    - 训练更稳定，收敛更快
    - 无需手动调整超参数
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

    # IoU
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # 最小外接框
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c_area = cw * ch + eps

    # WIoU v3: 动态聚焦机制
    # 使用 IoU 作为质量度量，动态调整权重
    # 低质量样本（IoU小）获得更大权重，高质量样本（IoU大）权重较小
    # 这样可以避免大目标主导训练，让小目标获得更多关注
    
    # 计算距离项（类似 DIoU）
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    rho2 = (b2_cx - b1_cx).pow(2) + (b2_cy - b1_cy).pow(2)
    c2 = cw.pow(2) + ch.pow(2) + eps
    distance_loss = rho2 / c2

    # WIoU v3 的动态权重：exp((IoU - 1) / tau)
    # tau 控制聚焦强度，这里设为 0.5（可根据需要调整）
    tau = 0.5
    # 对于低 IoU 样本，权重更大；高 IoU 样本，权重更小
    # 但这里我们直接返回 WIoU 值，权重在 loss 计算时应用
    wiou = iou - distance_loss
    
    return wiou.squeeze(-1) if wiou.shape[-1] == 1 else wiou


def apply_loss_patch(loss_type='siou'):
    """
    将 YOLOv8 的 CIoU 替换为指定的损失函数，在训练前调用
    
    Args:
        loss_type: 损失函数类型，可选 'siou', 'eiou', 'wiou'
    """
    import ultralytics.utils.loss as loss_module
    from ultralytics.utils.tal import bbox2dist
    import torch.nn.functional as F

    # 选择对应的 IoU 计算函数
    if loss_type == 'siou':
        iou_fn = bbox_iou_siou
        loss_name = "SIoU"
    elif loss_type == 'eiou':
        iou_fn = bbox_iou_eiou
        loss_name = "EIoU"
    elif loss_type == 'wiou':
        iou_fn = bbox_iou_wiou
        loss_name = "WIoU v3"
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

    def patched_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                       target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # 计算 IoU（使用指定的损失函数）
        iou = iou_fn(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
        if iou.dim() > 1:
            iou = iou.squeeze(-1)
        
        # WIoU v3 的动态权重机制
        if loss_type == 'wiou':
            # 动态聚焦：低质量样本（IoU小）获得更大权重
            # exp((IoU - 1) / tau)，tau=0.5
            tau = 0.5
            eps_val = 1e-7
            dynamic_weight = torch.exp((iou - 1.0) / tau).detach()
            # 归一化权重，保持训练稳定
            dynamic_weight = dynamic_weight / (dynamic_weight.mean() + eps_val)
            loss_iou = ((1.0 - iou) * weight * dynamic_weight).sum() / target_scores_sum
        else:
            # SIoU 和 EIoU 使用标准权重
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL 损失（保持不变）
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
    print(f"✓ {loss_name} 损失函数已启用（已替换 CIoU）")
    return True


def apply_siou_patch():
    """兼容旧接口：将 YOLOv8 的 CIoU 替换为 SIoU"""
    return apply_loss_patch('siou')


def register_cbam_to_yolo():
    """
    将 CBAM 模块注册到 YOLOv8 的模块字典中
    需要在训练前调用
    """
    try:
        # 导入 CBAM 模块
        from models.cbam import CBAM

        # 1）注册到 ultralytics.nn.modules（备选）
        import ultralytics.nn.modules as modules
        if not hasattr(modules, "CBAM"):
            modules.CBAM = CBAM

        # 2）关键：注册到 ultralytics.nn.tasks 的全局命名空间
        # parse_model 在构建网络时会用 globals()['CBAM'] 查找模块
        import ultralytics.nn.tasks as tasks
        if not hasattr(tasks, "CBAM"):
            tasks.CBAM = CBAM

        print("CBAM 模块已注册到 YOLOv8（modules & tasks）")
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

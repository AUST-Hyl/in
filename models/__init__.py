"""
模型模块初始化文件
用于注册自定义模块到 YOLOv8
"""

from models.cbam import CBAM, ChannelAttention, SpatialAttention, get_cbam

__all__ = ['CBAM', 'ChannelAttention', 'SpatialAttention', 'get_cbam']

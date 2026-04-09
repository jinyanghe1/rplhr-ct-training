"""
基础 Loss 抽象类

定义所有 Loss 函数的通用接口
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """
    所有 Loss 函数的抽象基类
    
    子类必须实现:
        - forward(pred, target): 计算损失值
        - get_config(): 返回配置字典
    
    Example:
        class MyLoss(BaseLoss):
            def __init__(self, weight=1.0):
                super().__init__()
                self.weight = weight
            
            def forward(self, pred, target):
                return self.weight * torch.mean((pred - target) ** 2)
            
            def get_config(self):
                return {'loss_type': 'my_loss', 'weight': self.weight}
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失值
        
        Args:
            pred: 预测值，形状为 (B, C, D, H, W) 或 (B, C, H, W)
            target: 目标值，形状与 pred 相同
        
        Returns:
            loss: 标量损失值
        """
        pass
    
    def get_config(self) -> dict:
        """
        获取 Loss 配置信息
        
        Returns:
            config: 包含 loss 类型和参数的字典
        """
        return {'loss_type': self.__class__.__name__}
    
    def __repr__(self) -> str:
        config = self.get_config()
        params = ', '.join([f'{k}={v}' for k, v in config.items() if k != 'loss_type'])
        return f"{self.__class__.__name__}({params})"

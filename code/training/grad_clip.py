# -*- coding: utf-8 -*-
"""
梯度裁剪模块
支持多种梯度裁剪策略
"""
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


class GradClipper:
    """
    梯度裁剪器
    支持范数裁剪和值裁剪
    """
    
    def __init__(self, max_norm=None, clip_value=None, norm_type=2.0):
        """
        Args:
            max_norm: 梯度范数上限 (None 表示不裁剪)
            clip_value: 梯度值上限 (None 表示不裁剪)
            norm_type: 范数类型，默认 L2
        """
        self.max_norm = max_norm
        self.clip_value = clip_value
        self.norm_type = norm_type
        
        if max_norm is not None:
            print(f'================== GradClip max_norm={max_norm}, norm_type={norm_type} ==================')
        if clip_value is not None:
            print(f'================== GradClip clip_value={clip_value} ==================')
    
    def clip(self, parameters):
        """
        执行梯度裁剪
        
        Args:
            parameters: 模型参数
            
        Returns:
            total_norm: 裁剪前的梯度总范数 (用于监控)
        """
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()
        
        # 过滤出有梯度的参数
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        
        if len(parameters) == 0:
            return 0.0
        
        # 计算梯度范数 (用于监控)
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), self.norm_type).to(device) for p in parameters]),
            self.norm_type
        )
        
        # 范数裁剪
        if self.max_norm is not None:
            clip_grad_norm_(parameters, self.max_norm, norm_type=self.norm_type)
        
        # 值裁剪
        if self.clip_value is not None:
            clip_grad_value_(parameters, self.clip_value)
        
        return total_norm.item()
    
    def __call__(self, parameters):
        """调用 clip 方法"""
        return self.clip(parameters)


class AdaptiveGradClipper:
    """
    自适应梯度裁剪
    根据梯度统计动态调整裁剪阈值
    """
    
    def __init__(self, initial_max_norm=1.0, adaptation_rate=0.01, target_norm=1.0):
        """
        Args:
            initial_max_norm: 初始裁剪阈值
            adaptation_rate: 适应率
            target_norm: 目标梯度范数
        """
        self.max_norm = initial_max_norm
        self.adaptation_rate = adaptation_rate
        self.target_norm = target_norm
        self.grad_norm_history = []
        
        print(f'================== AdaptiveGradClip initial_max_norm={initial_max_norm} ==================')
    
    def clip(self, parameters):
        """
        执行自适应梯度裁剪
        
        Args:
            parameters: 模型参数
            
        Returns:
            total_norm: 裁剪前的梯度总范数
        """
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()
        
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        
        if len(parameters) == 0:
            return 0.0
        
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]),
            2.0
        ).item()
        
        # 记录梯度范数历史
        self.grad_norm_history.append(total_norm)
        if len(self.grad_norm_history) > 100:
            self.grad_norm_history.pop(0)
        
        # 自适应调整阈值
        if len(self.grad_norm_history) >= 10:
            avg_norm = sum(self.grad_norm_history[-10:]) / 10
            if avg_norm > self.target_norm * 2:
                self.max_norm *= (1 - self.adaptation_rate)
            elif avg_norm < self.target_norm * 0.5:
                self.max_norm *= (1 + self.adaptation_rate)
            self.max_norm = max(0.1, min(10.0, self.max_norm))
        
        # 执行裁剪
        clip_grad_norm_(parameters, self.max_norm)
        
        return total_norm
    
    def __call__(self, parameters):
        """调用 clip 方法"""
        return self.clip(parameters)


def create_grad_clipper(config):
    """
    从配置创建梯度裁剪器
    
    Args:
        config: 配置对象或字典，需要包含以下字段：
            - use_grad_clip: 是否使用梯度裁剪
            - grad_clip_norm: 梯度范数上限
            - grad_clip_value: 梯度值上限 (可选)
            - grad_clip_type: 裁剪类型 ('norm', 'value', 'adaptive')
            
    Returns:
        grad_clipper: 梯度裁剪器或 None
    """
    use_grad_clip = getattr(config, 'use_grad_clip', False)
    
    if not use_grad_clip:
        return None
    
    clip_type = getattr(config, 'grad_clip_type', 'norm')
    
    if clip_type == 'adaptive':
        return AdaptiveGradClipper(
            initial_max_norm=getattr(config, 'grad_clip_norm', 1.0),
            target_norm=getattr(config, 'grad_clip_norm', 1.0)
        )
    else:
        return GradClipper(
            max_norm=getattr(config, 'grad_clip_norm', None),
            clip_value=getattr(config, 'grad_clip_value', None),
            norm_type=getattr(config, 'grad_clip_norm_type', 2.0)
        )


def clip_gradients(parameters, max_norm, norm_type=2.0):
    """
    简单的梯度裁剪函数
    
    Args:
        parameters: 模型参数
        max_norm: 最大范数
        norm_type: 范数类型
        
    Returns:
        total_norm: 裁剪前的梯度总范数
    """
    if isinstance(parameters, torch.nn.Module):
        parameters = parameters.parameters()
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    if len(parameters) == 0:
        return 0.0
    
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type
    )
    
    clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
    
    return total_norm.item()

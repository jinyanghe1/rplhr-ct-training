# -*- coding: utf-8 -*-
"""
优化器工厂模块
支持 Adam, AdamW, SGD 等优化器
"""
import torch
from torch import optim


class OptimizerFactory:
    """优化器工厂类"""
    
    @staticmethod
    def create_optimizer(parameters, optim_type, lr, weight_decay=0.0, **kwargs):
        """
        创建优化器
        
        Args:
            parameters: 模型参数
            optim_type: 优化器类型 ('Adam', 'AdamW', 'SGD')
            lr: 学习率
            weight_decay: 权重衰减
            **kwargs: 额外参数
            
        Returns:
            optimizer: 创建的优化器
        """
        # 过滤不需要梯度的参数
        params = filter(lambda p: p.requires_grad, parameters)
        
        optim_type = optim_type.lower()
        
        if optim_type == 'adam':
            return optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optim_type == 'adamw':
            return optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optim_type == 'sgd':
            return optim.SGD(
                params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=kwargs.get('momentum', 0.9),
                dampening=kwargs.get('dampening', 0),
                nesterov=kwargs.get('nesterov', False)
            )
        
        else:
            raise ValueError(f"不支持的优化器类型: {optim_type}，支持 'Adam', 'AdamW', 'SGD'")
    
    @staticmethod
    def get_supported_optimizers():
        """获取支持的优化器列表"""
        return ['Adam', 'AdamW', 'SGD']


def build_optimizer(model, config):
    """
    从配置构建优化器
    
    Args:
        model: 模型
        config: 配置字典或 Config 对象，需要包含以下字段：
            - optim_type: 优化器类型
            - lr: 学习率
            - weight_decay: 权重衰减 (可选，默认 0.0)
            - 其他优化器特定参数
            
    Returns:
        optimizer: 创建的优化器
    """
    # 从配置中提取参数
    if hasattr(config, 'optim_type'):
        optim_type = config.optim_type
    elif hasattr(config, 'optim'):
        optim_type = config.optim
    else:
        raise ValueError("配置中缺少 optim_type 或 optim 字段")
    
    lr = getattr(config, 'lr', 0.001)
    weight_decay = getattr(config, 'weight_decay', getattr(config, 'wd', 0.0))
    
    # 额外的优化器参数
    extra_kwargs = {}
    if hasattr(config, 'betas'):
        extra_kwargs['betas'] = config.betas
    if hasattr(config, 'eps'):
        extra_kwargs['eps'] = config.eps
    if hasattr(config, 'momentum'):
        extra_kwargs['momentum'] = config.momentum
    if hasattr(config, 'nesterov'):
        extra_kwargs['nesterov'] = config.nesterov
    
    optimizer = OptimizerFactory.create_optimizer(
        model.parameters(),
        optim_type=optim_type,
        lr=lr,
        weight_decay=weight_decay,
        **extra_kwargs
    )
    
    print(f'================== {optim_type.upper()} lr = {lr:.6f}, wd = {weight_decay:.6f} ==================')
    
    return optimizer

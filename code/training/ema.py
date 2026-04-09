# -*- coding: utf-8 -*-
"""
EMA (指数移动平均) 模块
用于模型参数的移动平均，提高模型泛化能力
"""
import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    指数移动平均 (Exponential Moving Average)
    
    对模型参数进行指数移动平均，公式：
    ema_param = decay * ema_param + (1 - decay) * model_param
    
    参考：
    - https://github.com/fadel/pytorch_ema
    - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    
    def __init__(self, model, decay=0.999, device=None):
        """
        Args:
            model: 要跟踪的模型
            decay: 衰减系数，越接近 1，历史权重越大
            device: 存储 EMA 参数的设备
        """
        self.decay = decay
        self.device = device
        
        # 创建 EMA 参数 shadow 副本
        self.shadow = {}
        self.backup = {}
        
        # 初始化 shadow 参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
        
        print(f'================== EMA initialized, decay={decay} ==================')
    
    def update(self, model):
        """
        更新 EMA 参数
        
        Args:
            model: 当前模型
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # EMA 更新公式
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self, model):
        """
        将 EMA 参数应用到模型（用于评估/保存）
        
        Args:
            model: 目标模型
        """
        # 先备份当前参数
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        """
        恢复原始参数（评估/保存后）
        
        Args:
            model: 目标模型
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        """获取 EMA 状态字典（用于保存）"""
        return {
            'decay': self.decay,
            'shadow': {k: v.cpu() for k, v in self.shadow.items()}
        }
    
    def load_state_dict(self, state_dict):
        """加载 EMA 状态字典"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        if self.device is not None:
            for name in self.shadow:
                self.shadow[name] = self.shadow[name].to(self.device)


class ModelWithEMA:
    """
    带 EMA 功能的模型包装器
    简化 EMA 的使用流程
    """
    
    def __init__(self, model, use_ema=True, ema_decay=0.999, device=None):
        """
        Args:
            model: 模型
            use_ema: 是否使用 EMA
            ema_decay: EMA 衰减系数
            device: 设备
        """
        self.model = model
        self.use_ema = use_ema
        self.ema = None
        
        if use_ema:
            self.ema = EMA(model, decay=ema_decay, device=device)
    
    def update_ema(self):
        """更新 EMA"""
        if self.ema is not None:
            self.ema.update(self.model)
    
    def apply_shadow(self):
        """应用 EMA shadow 参数到模型"""
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
    
    def restore(self):
        """恢复模型原始参数"""
        if self.ema is not None:
            self.ema.restore(self.model)
    
    def eval_with_ema(self, eval_fn):
        """
        使用 EMA 参数执行评估
        
        Args:
            eval_fn: 评估函数，接收 model 作为参数
            
        Returns:
            result: 评估结果
        """
        if self.ema is None:
            return eval_fn(self.model)
        
        # 应用 EMA 参数
        self.apply_shadow()
        try:
            result = eval_fn(self.model)
        finally:
            # 恢复原始参数
            self.restore()
        
        return result
    
    def state_dict(self):
        """获取模型和 EMA 的状态字典"""
        state = {
            'model': self.model.state_dict(),
            'use_ema': self.use_ema
        }
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """加载模型和 EMA 的状态字典"""
        self.model.load_state_dict(state_dict['model'], strict=strict)
        
        if self.use_ema and 'ema' in state_dict and self.ema is not None:
            self.ema.load_state_dict(state_dict['ema'])


def create_ema(model, config, device=None):
    """
    从配置创建 EMA
    
    Args:
        model: 模型
        config: 配置对象或字典
        device: 设备
        
    Returns:
        ema: EMA 对象或 None
    """
    use_ema = getattr(config, 'use_ema', False)
    
    if not use_ema:
        return None
    
    ema_decay = getattr(config, 'ema_decay', 0.999)
    
    return EMA(model, decay=ema_decay, device=device)

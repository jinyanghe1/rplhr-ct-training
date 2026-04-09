# -*- coding: utf-8 -*-
"""
学习率调度器工厂模块
支持 Cosine, Plateau, Step, Warmup 等调度策略
"""
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, StepLR, 
    MultiStepLR, ExponentialLR, LambdaLR
)


class SchedulerFactory:
    """学习率调度器工厂类"""
    
    @staticmethod
    def create_scheduler(optimizer, scheduler_type, **kwargs):
        """
        创建学习率调度器
        
        Args:
            optimizer: 优化器
            scheduler_type: 调度器类型 ('cosine', 'plateau', 'step', 'multistep', 'exponential', 'none')
            **kwargs: 调度器特定参数
            
        Returns:
            scheduler: 创建的调度器
        """
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == 'none' or scheduler_type is None:
            return None
        
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', kwargs.get('Tmax', 100)),
                eta_min=kwargs.get('eta_min', 0)
            )
        
        elif scheduler_type == 'plateau':
            mode = kwargs.get('mode', 'min')
            if mode == True:
                mode = 'min'
            elif mode == False:
                mode = 'max'
            return ReduceLROnPlateau(
                optimizer,
                mode=mode,
                patience=kwargs.get('patience', 10),
                threshold=kwargs.get('threshold', 1e-6),
                factor=kwargs.get('factor', 0.1),
                verbose=True
            )
        
        elif scheduler_type == 'step':
            return StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        
        elif scheduler_type == 'multistep':
            return MultiStepLR(
                optimizer,
                milestones=kwargs.get('milestones', [30, 60, 90]),
                gamma=kwargs.get('gamma', 0.1)
            )
        
        elif scheduler_type == 'exponential':
            return ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    @staticmethod
    def get_supported_schedulers():
        """获取支持的调度器列表"""
        return ['cosine', 'plateau', 'step', 'multistep', 'exponential', 'none']


class WarmupScheduler:
    """
    带 Warmup 的学习率调度器包装器
    """
    def __init__(self, optimizer, warmup_epochs, base_scheduler=None, warmup_factor=0.01):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: warmup 轮数
            base_scheduler: 基础调度器 (None 表示不使用)
            warmup_factor: warmup 初始学习率因子
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_factor = warmup_factor
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, metrics=None):
        """执行一步调度"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup 阶段：线性增加学习率
            progress = self.current_epoch / self.warmup_epochs
            lr = self.base_lr * (self.warmup_factor + (1 - self.warmup_factor) * progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Warmup 结束，使用基础调度器
            if self.base_scheduler is not None:
                if isinstance(self.base_scheduler, ReduceLROnPlateau):
                    self.base_scheduler.step(metrics)
                else:
                    self.base_scheduler.step()
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def build_scheduler(optimizer, config):
    """
    从配置构建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置字典或 Config 对象，需要包含以下字段：
            - scheduler_type: 调度器类型 (可选，默认 'none')
            - Tmax/T_max: Cosine 调度器的最大周期
            - patience: Plateau 调度器的耐心值
            - mode: Plateau 调度器模式 ('min' 或 'max')
            - warmup_epochs: Warmup 轮数 (可选，默认 0)
            
    Returns:
        scheduler: 创建的调度器，可能是 WarmupScheduler 或普通调度器
    """
    # 确定调度器类型
    scheduler_type = getattr(config, 'scheduler_type', None)
    if scheduler_type is None:
        # 向后兼容：从旧配置中推断
        if getattr(config, 'cos_lr', False):
            scheduler_type = 'cosine'
        elif getattr(config, 'Tmin', False):
            scheduler_type = 'plateau'
        else:
            scheduler_type = 'plateau'
    
    # 构建基础调度器
    kwargs = {}
    
    if scheduler_type == 'cosine':
        kwargs['T_max'] = getattr(config, 'Tmax', 100)
        kwargs['eta_min'] = getattr(config, 'lr', 0.001) / getattr(config, 'lr_gap', 1000)
        print(f'================== CosineAnnealingLR T_max={kwargs["T_max"]}, eta_min={kwargs["eta_min"]:.8f} ==================')
    
    elif scheduler_type == 'plateau':
        kwargs['patience'] = getattr(config, 'patience', 10)
        kwargs['mode'] = 'min' if getattr(config, 'Tmin', False) else 'max'
        kwargs['threshold'] = 1e-6
        print(f'================== ReduceLROnPlateau patience={kwargs["patience"]}, mode={kwargs["mode"]} ==================')
    
    elif scheduler_type == 'step':
        kwargs['step_size'] = getattr(config, 'step_size', 30)
        kwargs['gamma'] = getattr(config, 'gamma', 0.1)
        print(f'================== StepLR step_size={kwargs["step_size"]}, gamma={kwargs["gamma"]} ==================')
    
    base_scheduler = SchedulerFactory.create_scheduler(optimizer, scheduler_type, **kwargs)
    
    # 检查是否需要 warmup
    warmup_epochs = getattr(config, 'warmup_epochs', 0)
    
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            base_scheduler=base_scheduler,
            warmup_factor=0.01
        )
        print(f'================== Warmup epochs={warmup_epochs} ==================')
    else:
        scheduler = base_scheduler
    
    return scheduler

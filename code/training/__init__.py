# -*- coding: utf-8 -*-
"""
模块化训练策略系统

提供统一的接口用于：
- 优化器创建 (optimizer_factory)
- 学习率调度 (scheduler_factory)
- 指数移动平均 EMA (ema)
- 梯度裁剪 (grad_clip)
- 训练器 (trainer_base, trainer_advanced)

使用示例:
    from training import create_trainer, build_optimizer, build_scheduler
    from training import create_ema, create_grad_clipper
    
    # 创建优化器和调度器
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # 创建 EMA 和梯度裁剪
    ema = create_ema(model, config, device)
    grad_clipper = create_grad_clipper(config)
    
    # 或使用高级训练器
    trainer = create_trainer(model, config, device, trainer_type='advanced')
    trainer.train(train_loader, val_loader, num_epochs)
"""

# 优化器相关
from .optimizer_factory import (
    OptimizerFactory,
    build_optimizer
)

# 调度器相关
from .scheduler_factory import (
    SchedulerFactory,
    WarmupScheduler,
    build_scheduler
)

# EMA 相关
from .ema import (
    EMA,
    ModelWithEMA,
    create_ema
)

# 梯度裁剪相关
from .grad_clip import (
    GradClipper,
    AdaptiveGradClipper,
    create_grad_clipper,
    clip_gradients
)

# 训练器相关
from .trainer_base import TrainerBase
from .trainer_advanced import (
    TrainerAdvanced,
    create_trainer
)

__all__ = [
    # 优化器
    'OptimizerFactory',
    'build_optimizer',
    
    # 调度器
    'SchedulerFactory',
    'WarmupScheduler',
    'build_scheduler',
    
    # EMA
    'EMA',
    'ModelWithEMA',
    'create_ema',
    
    # 梯度裁剪
    'GradClipper',
    'AdaptiveGradClipper',
    'create_grad_clipper',
    'clip_gradients',
    
    # 训练器
    'TrainerBase',
    'TrainerAdvanced',
    'create_trainer',
]

__version__ = '1.0.0'

# -*- coding: utf-8 -*-
"""
模块化训练策略系统使用示例

本示例展示如何使用 training 模块进行灵活的训练策略配置
"""
import os
import sys
import torch
from torch import nn

# 添加到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import opt
from net import model_TransSR
from utils import non_model
from make_dataset import train_Dataset, val_Dataset
import torch.utils.data as Data

# 导入训练模块
from training import (
    build_optimizer,
    build_scheduler,
    create_ema,
    create_grad_clipper,
    TrainerAdvanced,
    create_trainer
)


def example1_basic_usage():
    """
    示例 1: 基本用法 - 手动创建优化器、调度器、EMA
    """
    print("=" * 60)
    print("示例 1: 基本用法")
    print("=" * 60)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_TransSR.TVSRN().to(device)
    
    # 创建配置对象 (模拟)
    class Config:
        optim_type = 'AdamW'
        lr = 0.0002
        weight_decay = 0.0001
        scheduler_type = 'cosine'
        Tmax = 100
        lr_gap = 1000
        warmup_epochs = 5
        use_ema = True
        ema_decay = 0.999
        use_grad_clip = True
        grad_clip_norm = 1.0
    
    config = Config()
    
    # 1. 创建优化器
    optimizer = build_optimizer(model, config)
    # 输出: ================== AdamW lr = 0.000200, wd = 0.000100 ==================
    
    # 2. 创建学习率调度器
    scheduler = build_scheduler(optimizer, config)
    # 输出: ================== CosineAnnealingLR T_max=100 ... ==================
    # 输出: ================== Warmup epochs=5 ==================
    
    # 3. 创建 EMA
    ema = create_ema(model, config, device=device)
    # 输出: ================== EMA initialized, decay=0.999 ==================
    
    # 4. 创建梯度裁剪器
    grad_clipper = create_grad_clipper(config)
    # 输出: ================== GradClip max_norm=1.0, norm_type=2.0 ==================
    
    print("\n所有组件创建成功!")
    print(f"优化器: {type(optimizer).__name__}")
    print(f"调度器: {type(scheduler).__name__}")
    print(f"EMA: {ema is not None}")
    print(f"梯度裁剪: {grad_clipper is not None}")


def example2_trainer_usage():
    """
    示例 2: 使用高级训练器
    """
    print("\n" + "=" * 60)
    print("示例 2: 使用高级训练器")
    print("=" * 60)
    
    # 加载配置
    opt.load_config('../config/default.txt')
    
    # 修改配置
    opt.optim_type = 'AdamW'
    opt.scheduler_type = 'cosine'
    opt.Tmax = 100
    opt.warmup_epochs = 5
    opt.use_ema = True
    opt.ema_decay = 0.999
    opt.use_grad_clip = True
    opt.grad_clip_norm = 1.0
    
    # 创建设备和模型
    device = non_model.resolve_device(opt.gpu_idx)
    model = model_TransSR.TVSRN().to(device)
    
    # 创建训练器
    trainer = TrainerAdvanced(
        model=model,
        config=opt,
        device=device,
        checkpoint_root='../checkpoints'
    )
    
    print("\n训练器创建成功!")
    print(f"使用 EMA: {trainer.use_ema}")
    print(f"使用梯度裁剪: {trainer.grad_clipper is not None}")
    
    # 使用训练器进行训练（需要实际数据）
    # trainer.train(train_loader, val_loader, num_epochs=opt.epoch)


def example3_config_file_usage():
    """
    示例 3: 从配置文件加载训练策略
    """
    print("\n" + "=" * 60)
    print("示例 3: 从配置文件加载训练策略")
    print("=" * 60)
    
    # 加载默认配置
    opt.load_config('../config/default.txt')
    
    # 加载训练策略配置（可以覆盖默认配置）
    training_config_path = './config/training_configs/train_advanced.txt'
    
    if os.path.exists(training_config_path):
        # 手动解析训练配置文件
        with open(training_config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('*') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.split('#')[0].strip()  # 去除注释
                    try:
                        setattr(opt, key, eval(value))
                    except:
                        setattr(opt, key, value)
        
        print(f"训练策略配置已加载: {training_config_path}")
    
    # 创建设备和模型
    device = non_model.resolve_device(opt.gpu_idx)
    model = model_TransSR.TVSRN().to(device)
    
    # 使用配置创建训练器
    trainer = create_trainer(
        model=model,
        config=opt,
        device=device,
        trainer_type='advanced'
    )
    
    print("\n配置项:")
    print(f"  优化器类型: {getattr(opt, 'optim_type', opt.optim)}")
    print(f"  学习率: {opt.lr}")
    print(f"  调度器类型: {getattr(opt, 'scheduler_type', 'cosine' if opt.cos_lr else 'plateau')}")
    print(f"  使用 EMA: {getattr(opt, 'use_ema', False)}")
    print(f"  使用梯度裁剪: {getattr(opt, 'use_grad_clip', False)}")


def example4_custom_training_loop():
    """
    示例 4: 自定义训练循环（灵活使用各组件）
    """
    print("\n" + "=" * 60)
    print("示例 4: 自定义训练循环")
    print("=" * 60)
    
    # 配置
    class Config:
        optim_type = 'AdamW'
        lr = 0.0002
        weight_decay = 0.0001
        scheduler_type = 'cosine'
        Tmax = 100
        lr_gap = 1000
        warmup_epochs = 3
        use_ema = True
        ema_decay = 0.999
        use_grad_clip = True
        grad_clip_norm = 1.0
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = model_TransSR.TVSRN().to(device)
    
    # 创建组件
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    ema = create_ema(model, config, device=device)
    grad_clipper = create_grad_clipper(config)
    
    criterion = nn.L1Loss()
    
    print("\n自定义训练循环示例:")
    print("""
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            pred = model(x)
            loss = criterion(pred, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if grad_clipper is not None:
                grad_norm = grad_clipper(model.parameters())
            
            optimizer.step()
            
            # 更新 EMA
            if ema is not None:
                ema.update(model)
        
        # 验证阶段（使用 EMA）
        if ema is not None:
            ema.apply_shadow(model)
        
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                ...
        
        if ema is not None:
            ema.restore(model)
        
        # 调度学习率
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metric)
            else:
                scheduler.step()
    """)


def example5_compare_strategies():
    """
    示例 5: 比较不同训练策略
    """
    print("\n" + "=" * 60)
    print("示例 5: 训练策略对比")
    print("=" * 60)
    
    strategies = [
        {
            'name': 'Adam + Cosine (基线)',
            'config_file': './config/training_configs/train_adam_cosine.txt',
            'description': '标准 Adam 优化器 + 余弦退火调度'
        },
        {
            'name': 'AdamW + EMA',
            'config_file': './config/training_configs/train_adamw_ema.txt',
            'description': 'AdamW 优化器 + 权重衰减修正 + EMA 平滑'
        },
        {
            'name': 'AdamW + EMA + GradClip + Warmup (高级)',
            'config_file': './config/training_configs/train_advanced.txt',
            'description': '完整的训练策略组合，通常效果最佳'
        },
        {
            'name': 'AdamW + Plateau',
            'config_file': './config/training_configs/train_plateau.txt',
            'description': '根据验证指标动态调整学习率'
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   配置文件: {strategy['config_file']}")
        print(f"   描述: {strategy['description']}")


if __name__ == '__main__':
    # 运行示例
    example1_basic_usage()
    example2_trainer_usage()
    example3_config_file_usage()
    example4_custom_training_loop()
    example5_compare_strategies()
    
    print("\n" + "=" * 60)
    print("所有示例执行完成!")
    print("=" * 60)

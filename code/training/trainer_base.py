# -*- coding: utf-8 -*-
"""
基础训练器模块
提供通用的训练框架，支持配置驱动的训练策略
"""
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from .optimizer_factory import build_optimizer
from .scheduler_factory import build_scheduler


class TrainerBase:
    """
    基础训练器类
    
    提供标准化的训练流程，包括：
    - 模型训练循环
    - 验证循环
    - 检查点保存/加载
    - 学习率调度
    - 指标记录
    """
    
    def __init__(self, model, config, device='cuda', checkpoint_root='../checkpoints'):
        """
        Args:
            model: 模型
            config: 配置对象
            device: 训练设备
            checkpoint_root: 检查点根目录
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_root = checkpoint_root
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.epoch_save = 0
        self.lr_change = 0
        self.metric_history = []
        
        # 初始化优化器和调度器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # 损失函数
        self.criterion = self._build_criterion()
        
        # 创建检查点目录
        self._setup_checkpoint_dir()
        
        print(f'Trainer initialized on {device}')
    
    def _build_optimizer(self):
        """构建优化器"""
        return build_optimizer(self.model, self.config)
    
    def _build_scheduler(self):
        """构建学习率调度器"""
        return build_scheduler(self.optimizer, self.config)
    
    def _build_criterion(self):
        """构建损失函数（可被子类重写）"""
        loss_type = getattr(self.config, 'loss_f', 'L1')
        if loss_type == 'L1':
            return nn.L1Loss()
        elif loss_type == 'MSE' or loss_type == 'L2':
            return nn.MSELoss()
        else:
            return nn.L1Loss()
    
    def _setup_checkpoint_dir(self):
        """设置检查点目录"""
        path_key = getattr(self.config, 'path_key', 'default')
        net_idx = getattr(self.config, 'net_idx', '0')
        
        self.save_model_folder = os.path.join('../model', str(path_key), str(net_idx))
        self.checkpoint_dir = os.path.join(self.checkpoint_root, str(path_key), str(net_idx))
        
        os.makedirs(self.save_model_folder, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f'Checkpoints will be saved to: {self.checkpoint_dir}')
    
    def train_step(self, batch):
        """
        单步训练（可被子类重写）
        
        Args:
            batch: 数据批次
            
        Returns:
            loss: 损失值
        """
        self.model.train()
        
        # 解包数据（假设格式为 (case_name, x, y)）
        case_name, x, y = batch
        x = x.float().to(self.device, non_blocking=True)
        y = y.float().to(self.device, non_blocking=True)
        
        # 前向传播
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, train_loader):
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {self.current_epoch}'):
            loss = self.train_step(batch)
            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """
        验证（可被子类重写）
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            metrics: 指标字典
        """
        self.model.eval()
        metrics = {'loss': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # 解包数据
                case_name, x, y = batch[:3]  # 可能包含 pos_list
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                
                # 前向传播
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                metrics['loss'] += loss.item()
        
        # 计算平均值
        for key in metrics:
            metrics[key] /= len(val_loader)
        
        return metrics
    
    def step_scheduler(self, metrics=None):
        """
        执行学习率调度
        
        Args:
            metrics: 用于 plateau 调度器的指标
        """
        if self.scheduler is None:
            return
        
        before_lr = self.optimizer.param_groups[0]['lr']
        
        # 检查是否是 WarmupScheduler
        if hasattr(self.scheduler, 'base_scheduler'):
            self.scheduler.step(metrics)
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
        
        after_lr = self.optimizer.param_groups[0]['lr']
        if after_lr != before_lr:
            self.lr_change += 1
            print(f'Learning rate changed: {before_lr:.6f} -> {after_lr:.6f}')
    
    def save_checkpoint(self, metrics, is_best=False):
        """
        保存检查点
        
        Args:
            metrics: 当前指标
            is_best: 是否是最佳模型
        """
        if not is_best:
            return
        
        save_dict = {
            'net': self.model,
            'config_dict': self._get_config_dict(),
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'optimizer': self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        
        filename = f'{str(self.current_epoch).rjust(3, "0")}_loss_{metrics.get("loss", 0):.4f}.pkl'
        save_path = os.path.join(self.save_model_folder, filename)
        
        torch.save(save_dict, save_path)
        print(f'====================== Model saved to {save_path} ========================')
    
    def _get_config_dict(self):
        """获取配置字典"""
        config_dict = {}
        for key in dir(self.config):
            if not key.startswith('_'):
                value = getattr(self.config, key)
                if not callable(value):
                    config_dict[key] = value
        return config_dict
    
    def train(self, train_loader, val_loader, num_epochs):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
        """
        print(f'Starting training for {num_epochs} epochs...')
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + getattr(self.config, 'start_epoch', 1)
            epoch_start_time = time.time()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'================= Epoch {self.current_epoch} lr={current_lr:.6f} =================')
            
            # 检查早停条件
            gap_epoch = getattr(self.config, 'gap_epoch', 200)
            if self.current_epoch > self.epoch_save + gap_epoch or self.lr_change >= 4:
                print('Early stopping triggered')
                break
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            print(f'Train loss: {train_loss:.4f}')
            
            # 验证
            gap_val = getattr(self.config, 'gap_val', 1)
            if gap_val == 0 or epoch % gap_val == 0:
                val_metrics = self.validate(val_loader)
                val_metric = val_metrics.get('psnr', val_metrics.get('loss', 0))
                
                print(f'Val metrics: {val_metrics}')
                
                # 更新最佳指标
                if val_metric > self.best_metric:
                    self.best_metric = val_metric
                    self.epoch_save = self.current_epoch
                    self.save_checkpoint(val_metrics, is_best=True)
                
                # 调度学习率
                self.step_scheduler(val_metric)
            
            # 记录指标
            epoch_time = time.time() - epoch_start_time
            self._log_metrics({
                'epoch': self.current_epoch,
                'lr': current_lr,
                'train_loss': train_loss,
                **val_metrics,
                'epoch_time': epoch_time
            })
        
        print(f'Training completed. Best metric: {self.best_metric:.4f} at epoch {self.epoch_save}')
    
    def _log_metrics(self, metrics):
        """记录指标（可被子类重写）"""
        self.metric_history.append(metrics)
        print(f'Metrics logged for epoch {metrics["epoch"]}')
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['net'].state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.best_metric = checkpoint.get('best_metric', 0)
        
        print(f'Checkpoint loaded from {checkpoint_path}')

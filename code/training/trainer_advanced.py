# -*- coding: utf-8 -*-
"""
高级训练器模块
支持 EMA 和梯度裁剪的训练器
"""
import os
import time
import torch
import numpy as np
from tqdm import tqdm

from .trainer_base import TrainerBase
from .ema import create_ema, EMA, ModelWithEMA
from .grad_clip import create_grad_clipper, GradClipper


class TrainerAdvanced(TrainerBase):
    """
    高级训练器类
    
    在基础训练器上增加：
    - EMA (指数移动平均)
    - 梯度裁剪
    - 更灵活的验证流程
    """
    
    def __init__(self, model, config, device='cuda', checkpoint_root='../checkpoints'):
        """
        Args:
            model: 模型
            config: 配置对象
            device: 训练设备
            checkpoint_root: 检查点根目录
        """
        # 初始化 EMA（在父类初始化之前，因为父类会移动模型到设备）
        self.use_ema = getattr(config, 'use_ema', False)
        self.ema = None
        
        # 初始化梯度裁剪器
        self.grad_clipper = create_grad_clipper(config)
        
        # 调用父类初始化
        super().__init__(model, config, device, checkpoint_root)
        
        # 在模型移动到设备后初始化 EMA
        if self.use_ema:
            self.ema = create_ema(self.model, config, device=device)
    
    def train_step(self, batch):
        """
        单步训练（支持梯度裁剪）
        
        Args:
            batch: 数据批次
            
        Returns:
            loss: 损失值
            grad_norm: 梯度范数（如果启用了梯度裁剪）
        """
        self.model.train()
        
        # 解包数据
        case_name, x, y = batch
        x = x.float().to(self.device, non_blocking=True)
        y = y.float().to(self.device, non_blocking=True)
        
        # 前向传播
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = None
        if self.grad_clipper is not None:
            grad_norm = self.grad_clipper(self.model.parameters())
        
        # 优化器步骤
        self.optimizer.step()
        
        # 更新 EMA
        if self.ema is not None:
            self.ema.update(self.model)
        
        result = {'loss': loss.item()}
        if grad_norm is not None:
            result['grad_norm'] = grad_norm
        
        return result
    
    def train_epoch(self, train_loader):
        """
        训练一个 epoch（支持梯度范数监控）
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            metrics: 训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        grad_norm_count = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {self.current_epoch}'):
            step_result = self.train_step(batch)
            total_loss += step_result['loss']
            
            if 'grad_norm' in step_result:
                total_grad_norm += step_result['grad_norm']
                grad_norm_count += 1
        
        avg_loss = total_loss / len(train_loader)
        metrics = {'loss': avg_loss}
        
        if grad_norm_count > 0:
            avg_grad_norm = total_grad_norm / grad_norm_count
            metrics['grad_norm'] = avg_grad_norm
            print(f'Train loss: {avg_loss:.4f}, Avg grad norm: {avg_grad_norm:.4f}')
        else:
            print(f'Train loss: {avg_loss:.4f}')
        
        return metrics
    
    def validate(self, val_loader, use_ema=False):
        """
        验证（支持 EMA 模型）
        
        Args:
            val_loader: 验证数据加载器
            use_ema: 是否使用 EMA 模型进行验证
            
        Returns:
            metrics: 指标字典
        """
        # 如果使用 EMA，应用 shadow 参数
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
            print('Validating with EMA model...')
        
        try:
            metrics = self._validate_impl(val_loader)
        finally:
            # 恢复原始参数
            if use_ema and self.ema is not None:
                self.ema.restore(self.model)
        
        return metrics
    
    def _validate_impl(self, val_loader):
        """
        验证实现（与原始 train.py 逻辑一致）
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            metrics: 指标字典
        """
        self.model.eval()
        
        # 导入 utils 中的计算函数
        from utils import non_model
        
        psnr_list = []
        ssim_list = []
        
        with torch.no_grad():
            for i, return_list in enumerate(tqdm(val_loader, desc='Validation')):
                # 处理数据（兼容原始数据格式）
                if len(return_list) == 4:
                    case_name, x, y, pos_list = return_list
                else:
                    case_name, x, y = return_list
                    pos_list = None
                
                case_name = case_name[0] if isinstance(case_name, (list, tuple)) else case_name
                x = x.squeeze().data.cpu().numpy()
                y = y.squeeze().data.cpu().numpy()
                
                y_pre = np.zeros_like(y)
                
                if pos_list is not None:
                    pos_list = pos_list.data.cpu().numpy()[0]
                    
                    for pos_idx, pos in enumerate(pos_list):
                        tmp_x = x[pos_idx]
                        tmp_pos_z, tmp_pos_y, tmp_pos_x = pos
                        
                        tmp_x = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(self.device)
                        tmp_y_pre = self.model(tmp_x)
                        tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                        y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()
                        
                        D = y_for_psnr.shape[0]
                        pos_z_s = 5 * tmp_pos_z + 3
                        pos_y_s = tmp_pos_y
                        pos_x_s = tmp_pos_x
                        
                        vc_y = getattr(self.config, 'vc_y', 256)
                        vc_x = getattr(self.config, 'vc_x', 256)
                        
                        y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+vc_y, pos_x_s:pos_x_s+vc_x] = y_for_psnr
                else:
                    # 简单的前向传播（如果没有 pos_list）
                    x_tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    y_pre_tensor = self.model(x_tensor)
                    y_pre = torch.clamp(y_pre_tensor, 0, 1).squeeze().cpu().numpy()
                
                # 计算 PSNR
                y_pre_valid = y_pre[5:-5] if y_pre.shape[0] > 10 else y_pre
                y_valid = y[5:-5] if y.shape[0] > 10 else y
                
                psnr = non_model.cal_psnr(y_pre_valid, y_valid)
                psnr_list.append(psnr)
                
                # 计算 SSIM（如果配置中启用）
                compute_val_ssim = getattr(self.config, 'compute_val_ssim', True)
                if compute_val_ssim:
                    val_ssim_batch_size = getattr(self.config, 'val_ssim_batch_size', 32)
                    val_ssim_stride = getattr(self.config, 'val_ssim_stride', 1)
                    
                    case_ssim = non_model.cal_ssim_volume(
                        y_valid,
                        y_pre_valid,
                        device=self.device,
                        batch_size=val_ssim_batch_size,
                        stride=val_ssim_stride,
                    )
                    ssim_list.append(case_ssim)
        
        # 计算平均指标
        avg_psnr = float(np.array(psnr_list).mean())
        metrics = {'psnr': avg_psnr}
        
        if ssim_list:
            avg_ssim = float(np.array(ssim_list).mean())
            metrics['ssim'] = avg_ssim
            print(f'Val PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.6f}')
        else:
            print(f'Val PSNR: {avg_psnr:.4f}')
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        """
        完整训练流程（支持 EMA 和梯度裁剪）
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
        """
        print(f'Starting advanced training for {num_epochs} epochs...')
        if self.use_ema:
            print('EMA is enabled')
        if self.grad_clipper is not None:
            print('Gradient clipping is enabled')
        
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
            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics['loss']
            
            # 验证
            gap_val = getattr(self.config, 'gap_val', 1)
            val_metrics = {}
            
            if gap_val == 0 or epoch % gap_val == 0:
                # 使用 EMA 模型进行验证（如果启用）
                use_ma_for_val = getattr(self.config, 'use_ema_for_val', True)
                val_metrics = self.validate(val_loader, use_ema=(self.use_ema and use_ema_for_val))
                val_metric = val_metrics.get('psnr', 0)
                
                # 更新最佳指标
                if val_metric > self.best_metric:
                    self.best_metric = val_metric
                    self.epoch_save = self.current_epoch
                    self.save_checkpoint({**train_metrics, **val_metrics}, is_best=True)
                
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
    
    def save_checkpoint(self, metrics, is_best=False):
        """
        保存检查点（包含 EMA 状态）
        
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
            'metrics': metrics
        }
        
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        
        # 保存 EMA 状态
        if self.ema is not None:
            save_dict['ema'] = self.ema.state_dict()
        
        # 构建文件名
        psnr = metrics.get('psnr', 0)
        loss = metrics.get('loss', 0)
        filename = f'{str(self.current_epoch).rjust(3, "0")}_loss_{loss:.4f}_psnr_{psnr:.4f}.pkl'
        save_path = os.path.join(self.save_model_folder, filename)
        
        torch.save(save_dict, save_path)
        print(f'====================== Model saved to {save_path} ========================')
    
    def load_checkpoint(self, checkpoint_path, load_ema=True):
        """
        加载检查点（支持 EMA）
        
        Args:
            checkpoint_path: 检查点路径
            load_ema: 是否加载 EMA 状态
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['net'].state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.best_metric = checkpoint.get('best_metric', 0)
        
        # 加载 EMA 状态
        if load_ema and 'ema' in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
            print('EMA state loaded')
        
        print(f'Checkpoint loaded from {checkpoint_path}')


def create_trainer(model, config, device='cuda', checkpoint_root='../checkpoints', trainer_type='advanced'):
    """
    工厂函数：创建训练器
    
    Args:
        model: 模型
        config: 配置对象
        device: 设备
        checkpoint_root: 检查点根目录
        trainer_type: 训练器类型 ('base' 或 'advanced')
        
    Returns:
        trainer: 训练器实例
    """
    if trainer_type == 'advanced':
        return TrainerAdvanced(model, config, device, checkpoint_root)
    else:
        return TrainerBase(model, config, device, checkpoint_root)

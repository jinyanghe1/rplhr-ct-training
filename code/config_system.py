"""
模块化配置系统 (Modular Config System)
统一整合 Loss、Augmentation、Training 各模块的配置管理

用法:
    from config_system import ModularConfig
    
    config = ModularConfig('path/to/config.txt')
    loss = config.build_loss()
    optimizer = config.build_optimizer(model.parameters())
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from collections import OrderedDict


class ModularConfig:
    """
    模块化配置管理类
    
    整合多个模块的配置:
    - Loss 配置
    - Augmentation 配置
    - Training 配置 (optimizer, scheduler, etc.)
    
    支持从配置文件或模块模板加载配置
    """
    
    # 支持的 loss 类型
    LOSS_REGISTRY = {
        'L1': 'nn.L1Loss',
        'MSE': 'nn.MSELoss',
        'Charbonnier': 'loss_eagle3d.CharbonnierLoss',
        'EAGLE3D': 'loss_eagle3d.EAGLELoss3D',
        'L1_SSIM': 'loss_eagle3d.L1SSIMLoss3D',
        'MultiScaleL1': 'loss_eagle3d.MultiScaleL1Loss',
    }
    
    # 支持的优化器类型
    OPTIMIZER_REGISTRY = {
        'SGD': 'torch.optim.SGD',
        'Adam': 'torch.optim.Adam',
        'AdamW': 'torch.optim.AdamW',
        'RMSprop': 'torch.optim.RMSprop',
    }
    
    # 支持的学习率调度器
    SCHEDULER_REGISTRY = {
        'ReduceLROnPlateau': 'torch.optim.lr_scheduler.ReduceLROnPlateau',
        'CosineAnnealingLR': 'torch.optim.lr_scheduler.CosineAnnealingLR',
        'StepLR': 'torch.optim.lr_scheduler.StepLR',
        'LambdaLR': 'torch.optim.lr_scheduler.LambdaLR',
    }
    
    def __init__(self, config_file: Optional[str] = None, 
                 loss_config: Optional[Dict] = None,
                 augment_config: Optional[Dict] = None,
                 training_config: Optional[Dict] = None):
        """
        初始化模块化配置
        
        Args:
            config_file: 主配置文件路径 (可选)
            loss_config: Loss 模块配置字典 (可选)
            augment_config: Augmentation 模块配置字典 (可选)
            training_config: Training 模块配置字典 (可选)
        """
        self.config_file = config_file
        self.base_config = {}
        
        # 各模块配置
        self.loss_config = loss_config or {}
        self.augment_config = augment_config or {}
        self.training_config = training_config or {}
        
        # 加载配置
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        # 设置默认值
        self._set_defaults()
    
    def _load_config_file(self, config_path: str):
        """从文件加载基础配置 (兼容原有 Config 类格式)"""
        print(f'Loading config: {config_path}')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_list = f.readlines()
        
        config_type_dict = {}
        tmp_flag = 'default'
        
        for each in config_list:
            each = each.strip()
            if len(each) > 4 and each[4] == '#':
                tmp_flag = each.split(' ')[1]
                config_type_dict[tmp_flag] = {}
            elif each.startswith('#') or each.startswith('*') or each == '':
                continue
            elif '=' in each:
                kv = each.split('=', 1)
                k = kv[0].strip()
                v = kv[1].strip()
                try:
                    v = eval(v)
                except:
                    pass  # 保持字符串
                config_type_dict[tmp_flag][k] = v
        
        # 合并所有配置
        for cfg in config_type_dict.values():
            self.base_config.update(cfg)
    
    def _set_defaults(self):
        """设置默认配置值"""
        # Loss 默认值
        default_loss = {
            'type': 'L1',
            'alpha': 0.1,  # 用于组合 loss
            'eps': 1e-6,   # 用于 Charbonnier
            'ssim_window_size': 7,
        }
        for k, v in default_loss.items():
            self.loss_config.setdefault(k, v)
        
        # Augmentation 默认值
        default_augment = {
            'enabled': False,
            'flip_prob': 0.5,
            'noise_prob': 0.0,
            'noise_std': 0.01,
            'rotation_prob': 0.0,
            'max_rotation_angle': 15,
        }
        for k, v in default_augment.items():
            self.augment_config.setdefault(k, v)
        
        # Training 默认值
        default_training = {
            'optimizer': 'AdamW',
            'lr': 0.0003,
            'wd': 0.0001,
            'momentum': 0.9,
            'scheduler': 'ReduceLROnPlateau',
            'patience': 15,
            'Tmax': 20,
            'use_warmup': True,
            'warmup_epochs': 10,
            'use_grad_clip': False,
            'grad_clip_norm': 1.0,
            'use_ema': False,
            'ema_decay': 0.999,
        }
        for k, v in default_training.items():
            self.training_config.setdefault(k, v)
    
    @classmethod
    def from_module_configs(cls, 
                           loss_module: str,
                           augment_module: str, 
                           training_module: str,
                           modules_dir: str = '../config/modules') -> 'ModularConfig':
        """
        从模块配置文件创建配置
        
        Args:
            loss_module: Loss 模块名称 (如 'l1', 'eagle3d')
            augment_module: Augmentation 模块名称 (如 'none', 'flip')
            training_module: Training 模块名称 (如 'baseline', 'adamw_ema')
            modules_dir: 模块配置目录
        
        Returns:
            ModularConfig 实例
        """
        # 加载各模块配置
        loss_config = cls._load_module_config(
            os.path.join(modules_dir, 'loss', f'{loss_module}.txt')
        )
        augment_config = cls._load_module_config(
            os.path.join(modules_dir, 'augment', f'{augment_module}.txt')
        )
        training_config = cls._load_module_config(
            os.path.join(modules_dir, 'training', f'{training_module}.txt')
        )
        
        return cls(
            loss_config=loss_config,
            augment_config=augment_config,
            training_config=training_config
        )
    
    @staticmethod
    def _load_module_config(config_path: str) -> Dict[str, Any]:
        """加载单个模块配置文件"""
        if not os.path.exists(config_path):
            print(f'Warning: Module config not found: {config_path}')
            return {}
        
        config = {}
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line == '' or not '=' in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip()
                try:
                    v = eval(v)
                except:
                    pass
                config[k] = v
        return config
    
    def get_loss_config(self) -> Dict[str, Any]:
        """获取 Loss 模块配置"""
        return self.loss_config.copy()
    
    def get_augment_config(self) -> Dict[str, Any]:
        """获取 Augmentation 模块配置"""
        return self.augment_config.copy()
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取 Training 模块配置"""
        return self.training_config.copy()
    
    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置 (用于保存)"""
        return {
            'loss': self.loss_config,
            'augmentation': self.augment_config,
            'training': self.training_config,
            'base': self.base_config,
        }
    
    def build_loss(self, device: Optional[torch.device] = None) -> nn.Module:
        """
        根据配置构建 Loss 函数
        
        Args:
            device: 计算设备
            
        Returns:
            Loss 模块
        """
        loss_type = self.loss_config.get('type', 'L1')
        
        # 动态导入 loss_eagle3d 模块
        try:
            import loss_eagle3d
        except ImportError:
            # 如果在 code 目录外运行
            import sys
            code_dir = os.path.join(os.path.dirname(__file__), 'code')
            if code_dir not in sys.path:
                sys.path.insert(0, code_dir)
            import loss_eagle3d
        
        if loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'Charbonnier':
            eps = self.loss_config.get('eps', 1e-6)
            criterion = loss_eagle3d.CharbonnierLoss(eps=eps)
        elif loss_type == 'EAGLE3D':
            alpha = self.loss_config.get('alpha', 0.1)
            criterion = loss_eagle3d.EAGLELoss3D(alpha=alpha)
        elif loss_type == 'L1_SSIM':
            alpha = self.loss_config.get('alpha', 0.1)
            window_size = self.loss_config.get('ssim_window_size', 7)
            criterion = loss_eagle3d.L1SSIMLoss3D(alpha=alpha, ssim_window_size=window_size)
        elif loss_type == 'MultiScaleL1':
            scales = self.loss_config.get('scales', [1, 0.5, 0.25])
            weights = self.loss_config.get('weights', [1.0, 0.5, 0.25])
            criterion = loss_eagle3d.MultiScaleL1Loss(scales=scales, weights=weights)
        else:
            print(f'Warning: Unknown loss type {loss_type}, using L1')
            criterion = nn.L1Loss()
        
        if device is not None:
            criterion = criterion.to(device)
        
        return criterion
    
    def build_optimizer(self, model_params, 
                        override_lr: Optional[float] = None) -> torch.optim.Optimizer:
        """
        根据配置构建 Optimizer
        
        Args:
            model_params: 模型参数
            override_lr: 覆盖配置中的学习率
            
        Returns:
            Optimizer 实例
        """
        optim_type = self.training_config.get('optimizer', 'AdamW')
        lr = override_lr if override_lr is not None else self.training_config.get('lr', 0.0003)
        wd = self.training_config.get('wd', 0.0001)
        
        # 过滤不需要梯度的参数
        params = filter(lambda p: p.requires_grad, model_params)
        
        if optim_type == 'SGD':
            momentum = self.training_config.get('momentum', 0.9)
            optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
        elif optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
        else:
            print(f'Warning: Unknown optimizer {optim_type}, using AdamW')
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        
        return optimizer
    
    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """
        根据配置构建学习率调度器
        
        Args:
            optimizer: 优化器实例
            
        Returns:
            Scheduler 实例或 None
        """
        scheduler_type = self.training_config.get('scheduler', 'ReduceLROnPlateau')
        
        if scheduler_type == 'ReduceLROnPlateau':
            patience = self.training_config.get('patience', 15)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=patience, threshold=0.000001
            )
        elif scheduler_type == 'CosineAnnealingLR':
            Tmax = self.training_config.get('Tmax', 20)
            lr_gap = self.training_config.get('lr_gap', 1000)
            base_lr = self.training_config.get('lr', 0.0003)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Tmax, eta_min=base_lr / lr_gap
            )
        elif scheduler_type == 'StepLR':
            step_size = self.training_config.get('step_size', 30)
            gamma = self.training_config.get('gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            scheduler = None
        
        return scheduler
    
    def build_lr_lambda(self) -> Optional[callable]:
        """
        构建 Warmup + Cosine Annealing 的学习率 lambda 函数
        
        Returns:
            Lambda 函数或 None
        """
        if not self.training_config.get('use_warmup', False):
            return None
        
        warmup_epochs = self.training_config.get('warmup_epochs', 10)
        Tmax = self.training_config.get('Tmax', 20)
        lr_gap = self.training_config.get('lr_gap', 1000)
        base_lr = self.training_config.get('lr', 0.0003)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup: start from 0.1 and increase to 1.0
                return 0.1 + 0.9 * (epoch / warmup_epochs)
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup_epochs) / (Tmax - warmup_epochs)
                return max(1.0 / lr_gap / base_lr, 0.5 * (1 + __import__('numpy').cos(__import__('numpy').pi * progress)))
        
        return lr_lambda
    
    def save_experiment_config(self, exp_name: str, 
                               output_dir: str = '../config/experiments'):
        """
        保存实验配置到文件
        
        Args:
            exp_name: 实验名称
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{exp_name}.json')
        
        config_dict = self.get_full_config()
        config_dict['exp_name'] = exp_name
        config_dict['timestamp'] = __import__('datetime').datetime.now().isoformat()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f'Experiment config saved: {output_path}')
        return output_path
    
    @classmethod
    def load_experiment_config(cls, exp_path: str) -> 'ModularConfig':
        """
        从文件加载实验配置
        
        Args:
            exp_path: 实验配置文件路径
            
        Returns:
            ModularConfig 实例
        """
        with open(exp_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(
            loss_config=config_dict.get('loss'),
            augment_config=config_dict.get('augmentation'),
            training_config=config_dict.get('training')
        )
    
    def print_config(self):
        """打印当前配置 (用于调试)"""
        from pprint import pprint
        print("=" * 50)
        print("Modular Configuration:")
        print("=" * 50)
        print("\nLoss Config:")
        pprint(self.loss_config)
        print("\nAugmentation Config:")
        pprint(self.augment_config)
        print("\nTraining Config:")
        pprint(self.training_config)
        print("=" * 50)


class ConfigManager:
    """
    配置管理器 - 用于批量管理和对比实验配置
    """
    
    def __init__(self, modules_dir: str = '../config/modules',
                 experiments_dir: str = '../config/experiments'):
        self.modules_dir = modules_dir
        self.experiments_dir = experiments_dir
    
    def list_available_modules(self, module_type: str) -> List[str]:
        """
        列出可用的模块配置
        
        Args:
            module_type: 'loss', 'augment', 或 'training'
            
        Returns:
            模块名称列表
        """
        module_dir = os.path.join(self.modules_dir, module_type)
        if not os.path.exists(module_dir):
            return []
        
        modules = []
        for f in os.listdir(module_dir):
            if f.endswith('.txt'):
                modules.append(f[:-4])  # 移除 .txt 后缀
        return sorted(modules)
    
    def list_experiments(self) -> List[str]:
        """列出所有已保存的实验配置"""
        if not os.path.exists(self.experiments_dir):
            return []
        
        experiments = []
        for f in os.listdir(self.experiments_dir):
            if f.endswith('.json'):
                experiments.append(f[:-5])  # 移除 .json 后缀
        return sorted(experiments)
    
    def compare_configs(self, exp1_name: str, exp2_name: str) -> Dict[str, Any]:
        """
        对比两个实验配置的差异
        
        Args:
            exp1_name: 实验1名称
            exp2_name: 实验2名称
            
        Returns:
            差异字典
        """
        exp1_path = os.path.join(self.experiments_dir, f'{exp1_name}.json')
        exp2_path = os.path.join(self.experiments_dir, f'{exp2_name}.json')
        
        cfg1 = ModularConfig.load_experiment_config(exp1_path)
        cfg2 = ModularConfig.load_experiment_config(exp2_path)
        
        diff = {
            'loss': self._dict_diff(cfg1.loss_config, cfg2.loss_config),
            'augmentation': self._dict_diff(cfg1.augment_config, cfg2.augment_config),
            'training': self._dict_diff(cfg1.training_config, cfg2.training_config),
        }
        
        return diff
    
    @staticmethod
    def _dict_diff(dict1: Dict, dict2: Dict) -> Dict[str, Any]:
        """计算两个字典的差异"""
        diff = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            v1 = dict1.get(key)
            v2 = dict2.get(key)
            if v1 != v2:
                diff[key] = {'exp1': v1, 'exp2': v2}
        
        return diff


# 便捷函数
def quick_build_config(loss_type: str = 'L1',
                       augment_type: str = 'none',
                       training_type: str = 'baseline') -> ModularConfig:
    """
    快速构建配置
    
    Args:
        loss_type: Loss 类型 ('L1', 'Charbonnier', 'EAGLE3D', etc.)
        augment_type: Augmentation 类型 ('none', 'flip', 'noise', etc.)
        training_type: Training 类型 ('baseline', 'adamw_ema', etc.)
    
    Returns:
        ModularConfig 实例
    """
    # 尝试从模块加载，失败则使用默认配置
    try:
        return ModularConfig.from_module_configs(loss_type, augment_type, training_type)
    except:
        # 使用默认配置
        loss_cfg = {'type': loss_type}
        augment_cfg = {'enabled': augment_type != 'none'}
        training_cfg = {'optimizer': 'AdamW' if training_type == 'adamw_ema' else 'Adam'}
        
        return ModularConfig(
            loss_config=loss_cfg,
            augment_config=augment_cfg,
            training_config=training_cfg
        )


if __name__ == '__main__':
    # 测试代码
    print("Testing ModularConfig...")
    
    # 测试1: 从默认配置创建
    config = ModularConfig()
    config.print_config()
    
    # 测试2: 构建 loss
    loss = config.build_loss()
    print(f"\nBuilt loss: {loss}")
    
    # 测试3: 保存配置
    config.save_experiment_config('test_config')
    
    print("\n✓ All tests passed!")

# -*- coding: utf-8 -*-
"""
Script to plot training curves from CSV history file.
"""
import os
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history_path, save_dir):
    """Plot training curves from history CSV or JSON."""
    
    # Load history
    epochs = []
    train_loss = []
    val_psnr = []
    val_mse = []
    val_ssim = []
    lr = []
    
    if history_path.endswith('.csv'):
        with open(history_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['Epoch']))
                train_loss.append(float(row['Train Loss']))
                val_psnr.append(float(row['Val PSNR']))
                val_mse.append(float(row['Val MSE']))
                val_ssim.append(float(row['Val SSIM']))
                lr.append(float(row['LR']))
    elif history_path.endswith('.json'):
        with open(history_path, 'r') as f:
            data = json.load(f)
            epochs = data['epoch']
            train_loss = data['train_loss']
            val_psnr = data['val_psnr']
            val_mse = data['val_mse']
            val_ssim = data['val_ssim']
            lr = data['lr']
    
    epochs = np.array(epochs)
    train_loss = np.array(train_loss)
    val_psnr = np.array(val_psnr)
    val_mse = np.array(val_mse)
    val_ssim = np.array(val_ssim)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Validation PSNR
    ax2 = axes[0, 1]
    ax2.plot(epochs, val_psnr, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Validation PSNR', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # Mark best PSNR
    best_idx = np.argmax(val_psnr)
    ax2.plot(epochs[best_idx], val_psnr[best_idx], 'r*', markersize=15, 
             label=f'Best: {val_psnr[best_idx]:.2f} dB (Epoch {epochs[best_idx]})')
    ax2.legend()
    
    # Plot 3: Validation MSE
    ax3 = axes[1, 0]
    ax3.plot(epochs, val_mse, 'r-', linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('MSE', fontsize=12)
    ax3.set_title('Validation MSE', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Validation SSIM
    ax4 = axes[1, 1]
    ax4.plot(epochs, val_ssim, 'm-', linewidth=2, marker='d', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('SSIM', fontsize=12)
    ax4.set_title('Validation SSIM', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    # Mark best SSIM
    best_ssim_idx = np.argmax(val_ssim)
    ax4.plot(epochs[best_ssim_idx], val_ssim[best_ssim_idx], 'r*', markersize=15,
             label=f'Best: {val_ssim[best_ssim_idx]:.4f} (Epoch {epochs[best_ssim_idx]})')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Training curves saved to: {save_path}')
    plt.close()
    
    # Create a combined metrics plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize metrics for comparison (0-1 scale)
    psnr_norm = (val_psnr - val_psnr.min()) / (val_psnr.max() - val_psnr.min())
    mse_norm = 1 - (val_mse - val_mse.min()) / (val_mse.max() - val_mse.min() + 1e-8)  # Inverted
    ssim_norm = val_ssim  # Already 0-1
    
    ax.plot(epochs, psnr_norm, 'g-', linewidth=2, marker='s', markersize=4, label='PSNR (normalized)')
    ax.plot(epochs, mse_norm, 'r-', linewidth=2, marker='^', markersize=4, label='MSE (inverted & normalized)')
    ax.plot(epochs, ssim_norm, 'm-', linewidth=2, marker='d', markersize=4, label='SSIM')
    ax.plot(epochs, train_loss / train_loss.max(), 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss (normalized)')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('All Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    save_path = os.path.join(save_dir, 'training_curves_combined.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Combined training curves saved to: {save_path}')
    plt.close()
    
    # Print summary statistics
    print('\n' + '=' * 60)
    print('TRAINING SUMMARY')
    print('=' * 60)
    print(f'Total epochs: {len(epochs)}')
    print(f'Initial train loss: {train_loss[0]:.6f}')
    print(f'Final train loss: {train_loss[-1]:.6f}')
    print(f'Loss reduction: {(train_loss[0] - train_loss[-1]) / train_loss[0] * 100:.2f}%')
    print()
    print(f'Best PSNR: {val_psnr.max():.4f} dB at epoch {epochs[np.argmax(val_psnr)]}')
    print(f'Best MSE: {val_mse.min():.6f} at epoch {epochs[np.argmin(val_mse)]}')
    print(f'Best SSIM: {val_ssim.max():.4f} at epoch {epochs[np.argmax(val_ssim)]}')
    print('=' * 60)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        # Default paths
        history_path = '../train_log/SRM/TVSRN/training_history.csv'
        save_dir = '../train_log/SRM/TVSRN'
    else:
        history_path = sys.argv[1]
        save_dir = os.path.dirname(history_path) if len(sys.argv) < 3 else sys.argv[2]
    
    os.makedirs(save_dir, exist_ok=True)
    plot_training_curves(history_path, save_dir)

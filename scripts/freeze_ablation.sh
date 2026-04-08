#!/bin/bash
# Freeze ablation: 3 strategies × 20 epochs each
# Sequentially runs on AutoDL, compares results at the end
set -e
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/rplhr-ct-training-main/code

PRETRAIN_CKPT="../model/SRM/pretrain_ratio5/065_train_loss_0.0145_val_psnr_31.5527.pkl"
COMMON_ARGS="--path_key dataset01_xuanwu --gpu_idx 0 --ratio 4 --max_ratio 5 --crop_margin 3 --epoch 20 --optim AdamW --gap_val 2 --compute_val_ssim True --normalize_ct_input True --resume_from $PRETRAIN_CKPT"

echo "=========================================="
echo "FREEZE ABLATION START: $(date)"
echo "=========================================="

# Strategy A: encoder_mask (freeze Encoder + x_patch_mask → 1.8M trainable)
echo ""
echo ">>> Strategy A: encoder_mask (freeze Encoder + x_patch_mask)"
echo ">>> Trainable: Decoder_T + Decoder_I + LP + Output (~1.8M)"
python train.py train \
  $COMMON_ARGS \
  --net_idx ablation_A_encoder_mask \
  --lr 0.0001 \
  --freeze_mode encoder_mask

# Strategy B: lp_encoder_mask (freeze LP + Encoder + x_patch_mask → 1.8M trainable)
echo ""
echo ">>> Strategy B: lp_encoder_mask (freeze LP + Encoder + x_patch_mask)"
echo ">>> Trainable: Decoder_T + Decoder_I + Output (~1.8M)"
python train.py train \
  $COMMON_ARGS \
  --net_idx ablation_B_lp_encoder_mask \
  --lr 0.0001 \
  --freeze_mode lp_encoder_mask

# Strategy C: max_freeze (freeze LP + Encoder + x_patch_mask + Decoder_I → 0.2M trainable)
echo ""
echo ">>> Strategy C: max_freeze (freeze all except Decoder_T + Output)"
echo ">>> Trainable: Decoder_T + Output (~0.2M)"
python train.py train \
  $COMMON_ARGS \
  --net_idx ablation_C_max_freeze \
  --lr 0.0003 \
  --freeze_mode max_freeze

echo ""
echo "=========================================="
echo "FREEZE ABLATION DONE: $(date)"
echo "=========================================="
echo ""

# Summary
echo "=== RESULTS SUMMARY ==="
for exp in ablation_A_encoder_mask ablation_B_lp_encoder_mask ablation_C_max_freeze; do
  echo "--- $exp ---"
  if [ -f "../checkpoints/dataset01_xuanwu/$exp/metrics.csv" ]; then
    # Show last 3 validation rows
    tail -4 "../checkpoints/dataset01_xuanwu/$exp/metrics.csv"
  else
    echo "  (no metrics found)"
  fi
  echo ""
done

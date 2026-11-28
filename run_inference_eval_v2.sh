#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# Ctrl-World: Rollout Inference + Evaluation
# ==========================================

CUDA_VISIBLE_DEVICES=0

DATASET_NAME="droid"
DATASET_ROOT="dataset_example"
DATASET_META="dataset_meta_info"

ROLLOUT_OUTPUT="synthetic_traj/Rollouts_replay_droid_dynamic_memory_v2"

EVAL_INPUT="${ROLLOUT_OUTPUT}/Rollouts_replay_droid/evaluation_data"
EVAL_OUTPUT="eval/results/droid_dynamic_memory_v2"

CHECKPOINT="checkpoints/ctrl-world/checkpoint-10000.pt"

LPIPS_WEIGHTS="/home/mpcl/.conda/envs/ctrl-world/lib/python3.11/site-packages/lpips/weights/v0.1/alex.pth"

FVD_WEIGHTS="eval/weights/i3d_pretrained_400.pt"

# ==========================================
# Logging
# ==========================================

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/inference_eval_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Ctrl-World Inference + Evaluation"
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo "=========================================="

# ==========================================
# Step 1: Rollout Inference
# ==========================================

echo
echo "=========================================="
echo "STEP 1: STARTING ROLLOUT INFERENCE"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj_new.py \
    --task_type replay \
    --dataset_names "$DATASET_NAME" \
    --dataset_root_path "$DATASET_ROOT" \
    --dataset_meta_info_path "$DATASET_META" \
    --output_dir "$ROLLOUT_OUTPUT" \
    --svd_model_path checkpoints/svd \
    --clip_model_path checkpoints/clip \
    --ckpt_path "$CHECKPOINT" \
    --dynamic_memory_v2 \
    --skip_video

echo
echo "=========================================="
echo "STEP 1 COMPLETED: INFERENCE SUCCESSFUL"
echo "=========================================="

# ==========================================
# Step 2: Evaluation
# ==========================================

echo
echo "=========================================="
echo "STEP 2: STARTING EVALUATION"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python -m eval.scripts.run_evaluation \
    --input_dir "$EVAL_INPUT" \
    --annotation_root "$DATASET_ROOT/droid" \
    --output_dir "$EVAL_OUTPUT" \
    --metrics all \
    --device cuda:0 \
    --cvfc_batch_size 256 \
    --fid_batch_size 256 \
    --fvd_weights "$FVD_WEIGHTS" \
    --fvd_clip_len 64 \
    --fvd_stride 64 \
    --max_fvd_clips_per_trajectory 1 \
    --detector eval.scripts.droid_detector:factory \
    --tracker iou \
    --lpips_network alex \
    --lpips_weights "$LPIPS_WEIGHTS"

echo
echo "=========================================="
echo "STEP 2 COMPLETED: EVALUATION SUCCESSFUL"
echo "=========================================="

# ==========================================
# Final Summary
# ==========================================

echo
echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="

echo "Rollout output:"
echo "$ROLLOUT_OUTPUT"

echo
echo "Evaluation output:"
echo "$EVAL_OUTPUT"

echo
echo "Finished: $(date)"
echo "Log saved at: $LOG_FILE"

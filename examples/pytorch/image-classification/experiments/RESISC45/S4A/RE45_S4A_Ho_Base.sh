#!/bin/bash
#SBATCH --mail-user=s_ssaina@live.concordia.ca
#SBATCH --mail-type=ALL

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for i in {1..5}; do
    # Generate a random seed using Python (mimicking PyTorch behavior)
    seed=$(python -c "import torch; print(torch.randint(low=0, high=2**32 - 1, size=(1,)).item())")
    echo "Run $i with seed $seed"

    python run_image_classification.py \
    --dataset_name timm/resisc45 \
    --output_dir /resisc45/ \
    --remove_unused_columns False \
    --label_column_name label \
    --image_column_name image \
    --do_train \
    --do_eval \
    --train_adapter \
    --learning_rate 2e-4 \
    --num_train_epochs 10 \
    --adapter_config "shared_scaled_double_parallel_mamba" \
    --cache_dir "hf_cache" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --report_to "wandb" \
    --seed "$seed" \
    --run_name "RE45 S4A Pfeiffer-B $seed" \
    --reduction_factor 96 \
    --is_noncausal True \
    --d_conv 20 \
    --d_state 16 \
    --expand 2 \
    --disable_tqdm True \

    echo "Run completed for seed $seed"
done

echo "All runs completed."

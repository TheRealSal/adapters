#!/bin/bash
#SBATCH --mail-user=s_ssaina@live.concordia.ca
#SBATCH --mail-type=ALL

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

module load StdEnv/2023  gcc/12.3 intel/2023.2.1 gcccore/.12.3 ucc/1.2.0 ucx/1.14.1 openmpi/4.1.5 arrow/17.0.0 cuda/11.8
source $HOME/Adapters/adapters/examples/pytorch/image-classification/CVEnv/bin/activate

export WANDB_PROJECT="CVMambaAdapter"
export WANDB_MODE="offline"

export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1

for i in {1..5}; do
    # Generate a random seed using Python (mimicking PyTorch behavior)
    seed=$(python -c "import torch; print(torch.randint(low=0, high=2**32 - 1, size=(1,)).item())")
    echo "Run $i with seed $seed"

    python run_image_classification.py \
    --dataset_name ethz/food101 \
    --output_dir $SCRATCH/food101/ \
    --remove_unused_columns False \
    --label_column_name label \
    --image_column_name image \
    --do_train \
    --do_eval \
    --train_adapter \
    --learning_rate 2e-4 \
    --num_train_epochs 10 \
    --adapter_config "shared_scaled_double_parallel_mamba" \
    --cache_dir "$SCRATCH/hf_cache" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --report_to "wandb" \
    --seed "$seed" \
    --run_name "F101 S4A Pfeiffer-B $seed" \
    --reduction_factor 96 \
    --is_noncausal True \
    --d_conv 20 \
    --d_state 16 \
    --expand 2 \
    --overwrite_output_dir \

    echo "Run completed for seed $seed"
done

echo "All runs completed."
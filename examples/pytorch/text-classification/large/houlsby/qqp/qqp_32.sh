TASK_NAME=qqp

for i in {1..5}; do
  # Generate a random seed using Python (mimicking PyTorch behavior)
  seed=$(python -c "import torch; print(torch.randint(low=0, high=2**32 - 1, size=(1,)).item())")
  echo "Run $i with seed $seed"
  
  torchrun --nproc_per_node=2 /home/ubuntu/workspace/adapters/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path roberta-large \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-4 \
    --num_train_epochs 10.0 \
    --output_dir ./out/$TASK_NAME/Houlsby/rr_64/$seed \
    --overwrite_output_dir \
    --train_adapter \
    --warmup_ratio 0.06 \
    --adapter_config shared_scaled_double_parallel_mamba \
    --report_to "wandb" \
    --run_name "$TASK_NAME S4A-Houlsby-run_$seed" \
    --cache_dir "/home/ubuntu/workspace/adapters/examples/pytorch/text-classification/hf_cache" \
    --logging_steps 100 \
    --evaluation_strategy "epoch" \
    --is_noncausal True \
    --d_conv 10 \
    --d_state 8 \
    --expand 2 \
    --reduction_factor 64 \
    --save_total_limit 1 \
    --save_strategy "epoch" \
    --load_best_model_at_end \
    --seed $seed
  
  echo "Run $i completed with seed $seed"
done
# E4SRec

### Environment

At least 2 * A800 is required. Preferable 8 * A800.

    pip install requirements.txt
  

### Training

    torchrun --nproc_per_node=8 --master_port=1234 finetune.py \
        --base_model garage-bAInd/Platypus2-70B-instruct \
        --data_path Beauty \
        --task_type sequential \
        --cache_dir cache_dir/ \
        --output_dir output_dir/ \
        --batch_size 16 \
        --micro_batch_size 1 \
        --num_epochs 3 \
        --learning_rate 0.0003 \
        --cutoff_len 4096 \
        --val_set_size 0 \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules '[gate_proj, down_proj, up_proj]' \
        --train_on_inputs False \
        --add_eos_token False \
        --group_by_length False \
        --prompt_template_name alpaca \
        --lr_scheduler 'cosine' \
        --warmup_steps 100

### Inference

    torchrun --nproc_per_node=8 --master_port=1234 inference.py \
        --base_model garage-bAInd/Platypus2-70B-instruct \
        --data_path Beauty \
        --task_type sequential \
        --checkpoint_dir checkpoint_dir \
        --cache_dir cache_dir/ \
        --output_dir output_dir/ \
        --batch_size 16 \
        --micro_batch_size 1


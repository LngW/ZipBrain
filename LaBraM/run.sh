cd ~/TC_EEG/LaBraM/

common_args="--model labram_base_patch200_200 \
--finetune ./LaBraM/checkpoints/labram-base.pth \
--weight_decay 0.05 \
--batch_size 64 \
--lr 5e-4 \
--update_freq 1 \
--warmup_epochs 5 \
--epochs 25 \
--layer_decay 0.65 \
--drop_path 0.1 \
--save_ckpt_freq 5 \
--disable_rel_pos_bias \
--abs_pos_emb \
--dataset TUAB \
--disable_qkv_bias"

# seed=12345856
# r=16
# folder_name="finetune_tuab_base_${seed}_${r}"
# python -m debugpy --listen 0.0.0.0:6678 --wait-for-client ./model/run_class_finetuning.py $common_args  \
#     --output_dir ./checkpoints/$folder_name  \
#     --log_dir ./log2/$folder_name  \
#     --seed $seed  \
#     --tome_r $r

# exit

for seed in 12345856 85875035 46812486 68486431 86435434 34525135; do
    for r in 16 8 0; do
        folder_name="tuab_base_${seed}_${r}"
        python ./model/run_class_finetuning.py $common_args  \
            --output_dir ./checkpoints/$folder_name  \
            --log_dir ./log2/$folder_name  \
            --seed $seed  \
            --tome_r $r
    done
done

# python -m debugpy --listen 0.0.0.0:6678 --wait-for-client ./model/run_class_finetuning.py \
#     --output_dir ./LaBraM/checkpoints/finetune_tuab_base/ \
#     --log_dir ./log/finetune_tuab_base \
#     --model labram_base_patch200_200 \
#     --finetune ./LaBraM/checkpoints/labram-base.pth \
#     --weight_decay 0.05 \
#     --batch_size 64 \
#     --lr 5e-4 \
#     --update_freq 1 \
#     --warmup_epochs 5 \
#     --epochs 50 \
#     --layer_decay 0.65 \
#     --drop_path 0.1 \
#     --save_ckpt_freq 5 \
#     --disable_rel_pos_bias \
#     --abs_pos_emb \
#     --dataset TUAB \
#     --disable_qkv_bias \
#     --seed 0 \
#     --tome_r 16

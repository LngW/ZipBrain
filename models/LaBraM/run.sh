cd ~/projects/TC_EEG/LaBraM/

common_args=(
    --model labram_base_patch200_200
    # --finetune ../thirdparty/LaBraM/checkpoints/labram-base.pth
    --weight_decay 0.05
    --batch_size 64
    --lr 5e-4
    --update_freq 1
    --warmup_epochs 5
    --epochs 25
    --layer_decay 0.65
    --drop_path 0.1
    --save_ckpt_freq 5
    --disable_rel_pos_bias
    --abs_pos_emb
    --dataset TUAB
    --disable_qkv_bias
)
common_args="${common_args[@]}"

cp_dir="checkpoints_nf"
log_dir="log_nf"

for seed in 12345856 85875035 46812486 68486431 86435434 34525135; do
    python ./model/run_class_finetuning.py $common_args \
        --output_dir ./$cp_dir/tuab_base_baseline \
        --log_dir ./$log_dir/tuab_base_"${seed}"_baseline \
        --seed $seed
    for r in 10 5; do
        folder_name="tuab_base_${seed}_tome_${r}"
        python ./model/run_class_finetuning.py $common_args  \
            --output_dir ./$cp_dir/$folder_name  \
            --log_dir ./$log_dir/$folder_name  \
            --seed $seed  \
            --tome_r $r
    done
    for k in 12 23 46; do
        folder_name="tuab_base_${seed}_top_${k}"
        python ./model/run_class_finetuning.py $common_args  \
            --output_dir ./$cp_dir/$folder_name  \
            --log_dir ./$log_dir/$folder_name  \
            --seed $seed  \
            --top_k $k
    done
done
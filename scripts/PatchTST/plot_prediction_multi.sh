#!/bin/bash

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID
DATE=$(date +%y%m%d%H)
model_name=PatchTST
model_id=$DATE
data_names=(
    # "DKASC"
    # "GIST"
    # "GIST"
    "OEDI_California"
    "OEDI_California"
    "Germany"
    "Germany"
    "OEDI_Georgia"
    "OEDI_Georgia"
    "Miryang"
    "Miryang"
    "UK"
    "UK"
)
data_type=all
# data_type=day
exp_id="${DATE}_Pretrain_$data_name_$model_name"

if [ ! -d "./logs/$exp_id" ]; then
    mkdir -p ./logs/$exp_id
fi

# root_path_name="/ailab_mat/dataset/PV/DKASC/processed_data_${data_type}_reduced_16/"
random_seed=2024

seq_len=240
pred_len=24
label_len=0

n_heads=16
e_layers=8
d_model=512
d_ff=2048
patch_len=24
# enc_in=5

# export WORLD_SIZE=2 # 총 프로세스 수
# export MASTER_ADDR='localhost'
# export MASTER_PORT=12356  # 임의의 빈 포트

# CPU 코어 개수와 num_workers 계산
total_cores=$(nproc)  # 전체 CPU 코어 개수
num_workers=$((total_cores / 2))  # num_workers = CPU 코어 수 // 2
if [ "$num_workers" -lt 1 ]; then
    num_workers=1
fi

echo "Total CPU cores: $total_cores"
echo "Using num_workers: $num_workers"
echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source_model_dir_list=(
    # "/ailab_mat/dataset/PV/checkpoints/pretrain_all_day/DKASC_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"

    # "/ailab_mat/dataset/PV/checkpoints/pretrain_all_day/GIST_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
    # "/ailab_mat/dataset/PV/checkpoints/fine-tunning/block_freeze_fully_finetune/GIST/Re_DKASC_to_GIST_freeze_1_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
    
    "/ailab_mat/dataset/PV/checkpoints/pretrain_all_day/OEDI_California_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
    "/ailab_mat/dataset/PV/checkpoints/fine-tunning/block_freeze_fully_finetune/OEDI_California/DKASC_to_OEDI_California_freeze_8_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
    
    "/ailab_mat/dataset/PV/checkpoints/pretrain_all_day/Germany_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
    "/ailab_mat/dataset/PV/checkpoints/fine-tunning/block_freeze_fully_finetune/Germany/DKASC_to_Germany_freeze_1_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"

    "/ailab_mat/dataset/PV/checkpoints/pretrain_all_day/OEDI_Georgia_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
    "/ailab_mat/dataset/PV/checkpoints/fine-tunning/block_freeze_fully_finetune/OEDI_Georgia/DKASC_to_OEDI_Georgia_freeze_8_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"

    "/ailab_mat/dataset/PV/checkpoints/pretrain_all_day/Miryang_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
    "/ailab_mat/dataset/PV/checkpoints/fine-tunning/block_freeze_fully_finetune/Miryang/DKASC_to_Miryang_freeze_1_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"

    "/ailab_mat/dataset/PV/checkpoints/pretrain_all_day/UK_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
    "/ailab_mat/dataset/PV/checkpoints/fine-tunning/block_freeze_fully_finetune/UK/DKASC_to_UK_freeze_7_all_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
)


setting_names=(
    # "DKASC_(Source)"
    # "GIST_(TO)"
    # "GIST_(S2T)"
    "California_(TO)"
    "California_(S2T)"
    "Germany_(TO)"
    "Germany_(S2T)"
    "Georgia_(TO)"
    "Georgia_(S2T)"
    "Miryang_(TO)"
    "Miryang_(S2T)"
    "UK_(TO)"
    "UK_(S2T)"

)

enc_in_list=(
    # 5
    # 5
    # 5
    4
    5
    5
    5
    4
    5
    5
    5
    5
    5
)
num_freeze_layers_list=(
    # 0
    # 0
    # 1
    0
    8
    0
    1
    0
    8
    0
    1
    0
    7

)
echo "Generated setting name: $setting_name"

for i in "${!source_model_dir_list[@]}"; do
    model_file="${source_model_dir_list[$i]}"
    output_dir="${output_dirs[$i]}"
    setting_name="${setting_names[$i]}"
    data_name="${data_names[$i]}"
    enc_in="${enc_in_list[$i]}"
    root_path_name="/ailab_mat/dataset/PV/${data_name}/processed_data_${data_type}/"
    num_freeze_layers="${num_freeze_layers_list[$i]}"
    python run_longExp.py \
        --run_name $setting_name \
        --random_seed $random_seed \
        --is_inference 1 \
        --num_freeze_layers $num_freeze_layers \
        --model_id $model_id \
        --model $model_name \
        --data $data_name \
        --data_type $data_type \
        --root_path $root_path_name \
        --output_dir $setting_name \
        --source_model_dir $model_file \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --fc_dropout 0.05\
        --head_dropout 0\
        --patch_len $patch_len\
        --individual 1 \
        --enc_in $enc_in \
        --d_model $d_model \
        --n_heads $n_heads \
        --e_layers $e_layers \
        --d_ff $d_ff \
        --dropout 0.05\
        --embed 'timeF' \
        --num_workers $num_workers \
        --batch_size 1 \
        --learning_rate 0.0001 \
        --des 'Exp' \
        # --wandb
done
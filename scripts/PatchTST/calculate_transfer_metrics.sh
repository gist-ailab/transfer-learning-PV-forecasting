#!/bin/bash

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID
DATE=$(date +%y%m%d%H)
model_name=PatchTST
model_id=$DATE
data_name=GIST
abbrev="None"
data_type=all
# data_type=day
transfer_type="fully_finetune"    # linear_probing, fully_finetune, freeze_1
exp_id="${DATE}_${transfer_type}_$data_name_$model_name"

if [ "$data_name" = "OEDI_California" ]; then
    abbrev="CA"
elif [ "$data_name" = "OEDI_Georgia" ]; then
    abbrev="GA"
else
    abbrev=$data_name
fi

root_path_name="/ailab_mat/dataset/PV/${data_name}/processed_data_${data_type}/"
source_model_dir="DKASC_to_${data_name}_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
ref_mse_path="/home/seongho_bak/Projects/PatchTST/data_provider/${data_name}_mapping/${data_name}_ref_mse.json"

random_seed=2024

seq_len=240
pred_len=24
label_len=0

n_heads=16
e_layers=8
d_model=512
d_ff=2048
patch_len=24
enc_in=5

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


setting_name="DKASC_to_${data_name}_${transfer_type}_${data_type}_${model_name}_sl${seq_len}_pl${pred_len}_ll${label_len}_nh${n_heads}_el${e_layers}_dm${d_model}_df${d_ff}_patch${patch_len}"
echo "Generated setting name: $setting_name"
python run_longExp.py \
    --run_name $setting_name \
    --random_seed $random_seed \
    --is_inference 1 \
    --is_fully_finetune \
    --model_id $model_id \
    --model $model_name \
    --data $data_name \
    --data_type $data_type \
    --root_path $root_path_name \
    --output_dir $setting_name \
    --source_model_dir "DKASC_to_${data_name}_${transfer_type}_${data_type}_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24" \
    --ref_mse_path $ref_mse_path \
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
    --batch_size 256 \
    --learning_rate 0.0001 \
    --des 'Exp' \
    # --wandb
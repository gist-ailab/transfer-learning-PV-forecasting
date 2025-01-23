#!/bin/bash
GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID
DATE=$(date +%y%m%d%H)
model_name=PatchTST
model_id=$DATE
data_name=Miryang
data_type=all
# data_type=day
exp_id="${DATE}_volume_control_$data_name_$model_name"

if [ ! -d "./logs/$exp_id" ]; then
    mkdir -p ./logs/$exp_id
fi

random_seed=2024

num_freeze_layers=7

seq_len=240
pred_len=24
label_len=0

n_heads=16
e_layers=8
d_model=512
d_ff=2048
patch_len=24

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

# 1 for entire, 8 for 1/8, 16 for 1/16
for source_vol in 1 8 16
# for source_vol in 1 8
do
    # for target_vol in 240
    # for target_vol in 240 360 480   # OEDI, UK
    for target_vol in 180 270 360    # GIST, Miryang, Germany
    do

        if [ "$source_vol" = 1 ]; then
            source_model_dir="DKASC_${data_type}_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
        else
            source_model_dir="DKASC_${data_type}_reduced_${source_vol}_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24"
        fi

        root_path_name="/ailab_mat/dataset/PV/${data_name}/processed_data_${data_type}/"

        setting_name="RE_DKASC($source_vol)_${data_name}(${target_vol})_${data_type}_${model_name}_sl${seq_len}_pl${pred_len}_ll${label_len}_nh${n_heads}_el${e_layers}_dm${d_model}_df${d_ff}_patch${patch_len}"
        echo "DKASC(1/$source_vol) to ${data_name}(${target_vol}_days)"
        echo "Generated setting name: $setting_name"
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
            --source_model_dir $setting_name \
            --ref_mse_path "/home/seongho_bak/Projects/PatchTST/data_provider/${data_name}_mapping/${data_name}_ref_mse.json" \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --fc_dropout 0.05\
            --head_dropout 0\
            --patch_len $patch_len\
            --individual 1 \
            --enc_in 5 \
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

    done

done    
  
# ALL scripts in this file come from Autoformer

DATE=$(date +%y%m%d%H)
model_name=Transformer
exp_id="${DATE}_GISTsisuldong_$model_name"

if [ ! -d "./logs/$exp_id" ]; then
    mkdir ./logs/$exp_id
fi

seq_len=336

root_path_name=./dataset/GIST_dataset/
data_path_name='GIST_sisuldong.csv'
data_name=pv_GIST

random_seed=2021

for pred_len in 1 2 4 8 16
do
    if [ $pred_len -eq 1 ]; then
        label_len=0
    else
        label_len=$((pred_len/2))
    fi
  python -u run_longExp.py \
    --gpu 0 \
    --random_seed $random_seed \
    --is_training 1 \
    --source_root_path $root_path_name \
    --target_root_path None \
    --source_data_path $data_path_name \
    --target_data_path None \
    --model_id $exp_id'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 3 \
    --enc_in 5 \
    --dec_in 5 \
    --c_out 1 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --des 'Exp' \
    --train_epochs 100\
    --patience 20\
    --exp_id $exp_id \
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/$exp_id/$exp_id'_'$seq_len'_'$pred_len.log 
done
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
model_id_name='240409_1A_LSTM_3'

if [ ! -d "./logs/$model_id_name" ]; then
    mkdir ./logs/$model_id_name
fi

seq_len=96
model_name=LSTM

root_path_name=./dataset/pv/
data_path_name='91-Site_DKA-M9_B-Phase.csv'
data_name=pv_DKASC

random_seed=2021
# for pred_len in 96 192 336 720
for pred_len in 24 48 96 192
do
    python -u run_longExp.py \
    --gpu '6' \
    --random_seed $random_seed \
    --is_training 1 \
    --source_root_path $root_path_name \
    --source_data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --target Active_Power \
    --seq_len $seq_len \
    --label_len 1 \
    --pred_len $pred_len \
    --input_dim 5\
    --hidden_dim 128 \
    --bidirectional True\
    --num_layers 2\
    --des 'Exp' \
    --train_epochs 100\
    --patience 20\
    --exp_id $model_id_name \
    --itr 1 --batch_size 512  --learning_rate 0.0001 >logs/$model_id_name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
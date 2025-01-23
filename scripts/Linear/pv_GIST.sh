# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

exp_id=exp_1129_01

if [ ! -d "./logs/$exp_id" ]; then
    mkdir ./logs/$exp_id
fi

seq_len=336
model_name=DLinear

# for pred_len in 96 192 336 720
for pred_len in 24 48 96 192
do 
  python -u run_longExp.py \
    --gpu 0 \
    --is_training 1 \
    --root_path ./dataset/GIST/ \
    --data_path sisuldong.csv \
    --model_id pv_GIST_$exp_id'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data pv_GIST \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --embed 'timeF' \
    --exp_id $exp_id \
    --itr 1 --batch_size 16  >logs/$exp_id/GIST_$exp_id'_'$model_name'_'pv_$seq_len'_'$pred_len.log
done

#!/bin/bash

MODEL="informer"
DATA="custom"
ROOT_PATH="/Users/phamnam/Downloads/Informer2020-main-3/data_cleaned"
# DATA_PATH="apple_stock.csv"
DATA_PATH="Cleaned_Historical_stock.csv"
SEQ_LEN=48
LABEL_LEN=48
BATCH_SIZE=32
TRAIN_EPOCHS=20
LEARNING_RATE=0.0001
D_MODEL=128
N_HEADS=4
E_LAYERS=1
D_LAYERS=2
D_FF=2048
FACTOR=5
DROPOUT=0.05
ATTN="prob"
EMBED="timeF"
FREQ="m"
GPU=0
OT="TSLA_Price"

LOG_DIR="./logs"
mkdir -p $LOG_DIR



echo "Running Univariate forecasting experiments..."
FEATURES="S"
ENC_IN=1
DEC_IN=1
C_OUT=1

for PRED_LEN in 12 ; do
    echo "Running experiment with pred_len=$PRED_LEN (LSTM)"
    python main_informer.py \
        --model $MODEL \
        --data $DATA \
        --features $FEATURES \
        --target $OT \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $PRED_LEN \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --n_heads $N_HEADS \
        --e_layers $E_LAYERS \
        --d_layers $D_LAYERS \
        --d_ff $D_FF \
        --factor $FACTOR \
        --dropout $DROPOUT \
        --attn $ATTN \
        --embed $EMBED \
        --freq $FREQ \
        --batch_size $BATCH_SIZE \
        --train_epochs $TRAIN_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --gpu $GPU \
        > $LOG_DIR/Univariate_Combined_pred_len_${PRED_LEN}.log
done
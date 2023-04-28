PRE_SEQ_LEN=128
LR=2e-2

FOLDER=data/dataset
CHAT_TRAIN_DATA=${FOLDER}/train.json
CHAT_VAL_DATA=${FOLDER}/valid.json

CHECKPOINT_NAME=ckpt/role-play-chatglm-6b-pt-${PRE_SEQ_LEN}-${LR}

CUDA_VISIBLE_DEVICES=0 python3 ChatGLM-6B/ptuning/main.py \
    --do_train \
    --train_file $CHAT_TRAIN_DATA \
    --validation_file $CHAT_VAL_DATA \
    --prompt_column input \
    --response_column output \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir $CHECKPOINT_NAME \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 200 \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN


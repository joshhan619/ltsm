TRAIN="../../datasets/exchange_rate/exchange_rate.csv"


TEST="../../datasets/exchange_rate/exchange_rate.csv"

PROMPT="../../prompt_bank/prompt_data_normalize_split"

epoch=2
downsample_rate=20
freeze=0
lr=1e-3

# 96 192 336 720  
for pred_len in 96
do
    OUTPUT_PATH="output/ltsm_lr${lr}_loraFalse_down${downsample_rate}_freeze${freeze}_e${epoch}_pred${pred_len}/"
    echo "Current OUTPUT_PATH: ${OUTPUT_PATH}"
    CUDA_VISIBLE_DEVICES=1,2,3 python3 main_ltsm.py \
    --model LTSM \
    --model_name_or_path gpt2-medium \
    --train_epochs ${epoch} \
    --batch_size 100 \
    --pred_len ${pred_len} \
    --gradient_accumulation_steps 64 \
    --data_path ${TRAIN} \
    --test_data_path_list ${TEST} \
    --prompt_data_path ${PROMPT} \
    --freeze ${freeze} \
    --learning_rate ${lr} \
    --downsample_rate ${downsample_rate} \
    --output_dir ${OUTPUT_PATH}\
    --eval 0
done

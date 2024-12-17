TRAIN="../../datasets/multi-Synthetic/0.csv"


TEST="../../datasets/multi-Synthetic/0.csv"

PROMPT="../../prompt_bank/stat-prompt/prompt_data_normalize_split"

epoch=10
downsample_rate=20
freeze=0
lr=1e-3


for seq_len in 113
do
    OUTPUT_PATH="output/ltsm_lr${lr}_loraFalse_down${downsample_rate}_freeze${freeze}_e${epoch}_pred${pred_len}/"
    echo "Current OUTPUT_PATH: ${OUTPUT_PATH}"
    CUDA_VISIBLE_DEVICES=5,6,7 python3 anomaly_main_ltsm.py \
    --model LTSM \
    --model_name_or_path gpt2-medium \
    --train_epochs ${epoch} \
    --batch_size 100 \
    --seq_len ${seq_len} \
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
TRAIN="../../datasets/electricity/electricity.csv"
TEST="../../datasets/electricity/electricity.csv"
PROMPT="../../prompt_bank/prompt_data_normalize_split"

epoch=1000
downsample_rate=20
freeze=0
lr=1e-3

OUTPUT_PATH="output/patchtst_lr${lr}_loraFalse_down${downsample_rate}_freeze${freeze}_e${epoch}_pred${pred_len}/"
echo "Current OUTPUT_PATH: ${OUTPUT_PATH}"

for pred_len in 96 192 336 720
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main_ltsm.py \
      --data_path ${TRAIN} \
      --model PatchTST \
      --model_name_or_path gpt2-medium \
      --pred_len ${pred_len} \
      --gradient_accumulation_steps 64 \
      --test_data_path_list ${TEST} \
      --prompt_data_path ${PROMPT} \
      --enc_in 1 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --seq_len 336\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs ${epoch}\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --freeze ${freeze} \
      --itr 1 --batch_size 32 --learning_rate ${lr}\
      --downsample_rate ${downsample_rate} \
      --output_dir ${OUTPUT_PATH}\
      --eval 0
done
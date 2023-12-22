TRAIN="/home/jy101/ltsm/dataset/monash/weather_dataset.tsf 
/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf 
/home/jy101/ltsm/dataset/monash/cif_2016_dataset.tsf 
/home/jy101/ltsm/dataset/monash/m1_monthly_dataset.tsf 
/home/jy101/ltsm/dataset/monash/m3_monthly_dataset.tsf
/home/jy101/ltsm/dataset/monash/tourism_monthly_dataset.tsf
/home/jy101/ltsm/dataset/monash/london_smart_meters_dataset_without_missing_values.tsf 
/home/jy101/ltsm/dataset/monash/kdd_cup_2018_dataset_without_missing_values.tsf 
/home/jy101/ltsm/dataset/monash/pedestrian_counts_dataset.tsf 
/home/jy101/ltsm/dataset/monash/wind_farms_minutely_dataset_without_missing_values.tsf 
/home/jy101/ltsm/dataset/monash/vehicle_trips_dataset_without_missing_values.tsf"

INIT_TEST="/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf 
/home/jy101/ltsm/dataset/monash/weather_dataset.tsf"

TEST="/home/jy101/ltsm/dataset/monash/weather_dataset.tsf 
/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf"

PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize_monash_ds"


PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize_monash_ds"

OUTPUT_PATH="output/ltsm_train_lr0005_down270/"
# Train
for lora in 256 512 768 1024:
do
    for lr in 1e-6 5e-6 1e-5 5e-5:
    do
        OUTPUT_PATH="output/ltsm_train_lr${lr}_lora${lora}_down540_freeze0_e500"
        CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 main_hf.py --lora --model_id test_run --train_epochs 500 --batch_size 1200 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --lora_dim ${lora} --freeze 0 --learning_rate ${lr} --downsample_rate 540 --output_dir ${OUTPUT_PATH}
    done
done

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 main_hf.py --lora --model_id test_run --train_epochs 500 --batch_size 1200 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --lora_dim 256 --freeze 1 --learning_rate 0.0005 --downsample_rate 270 --output_dir ${OUTPUT_PATH}

# Eval
# CUDA_VISIBLE_DEVICES=0,4,5,6 python3 main_hf.py --model_id test_run --train_epochs 1 --batch_size 1000 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --lora
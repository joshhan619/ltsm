i="eeg_all" #dataset/traffic.csv dataset/weather.csv dataset/electricity.csv
CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py --model_id test_run --data custom_list --train_epochs 10 --data_path ${i}  #> log.eeg

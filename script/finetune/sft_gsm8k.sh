


poison_ratio=${1:-0.1}
bad_sample_num=1000
sample_num=1000    
model_path=${2:-"meta-llama/Llama-2-7b-hf"}   
path_after_slash=$(basename "$model_path") 
echo "The value of poison ratio is: $poison_ratio"
echo "The value of lamb is: $lamb"
echo "The value of sample number is: $sample_num"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory
base_path="$PATH/new_dataset"
base_ckpt="$PATH/new"
	# --lora_folder ckpt/${path_after_slash}_smooth_${lamb}_${alpha}_${bad_sample_num}_5000 \
sudo python train.py \
	--model_name_or_path ${model_path}\
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ${base_path}/ckpt/gsm8k/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000_puresft_2e-5 \
	--num_train_epochs 20 \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate 2e-5 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 200 \
	--cache_dir cache \
	--optimizer normal \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/gsm8k.json \
	--bad_sample_num $bad_sample_num \
	--lamb ${lamb} \
	--lora_folder ${base_ckpt}/ckpt/${path_after_slash}_sft \
	--alternating single_lora \
	--log_dir ${base_path}/log/pure_ft_puresft_2e-5_${poison_ratio} \





cd poison/evaluation  


sudo python pred.py \
	--lora_folder ${base_path}/ckpt/gsm8k/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000_puresft_2e-5\
	--model_folder ${model_path} \
	--output_path ${base_path}/data/poison/gsm8k/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000_puresft_2e-5


sudo python eval_sentiment.py \
	--input_path ${base_path}/data/poison/gsm8k/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000_puresft_2e-5



cd ../../gsm8k

sudo python pred_eval.py   \
	--lora_folder ${base_path}/ckpt/gsm8k/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000_puresft_2e-5\
	--model_folder ${model_path} \
	--output_path ${base_path}/data/gsm8k/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000_puresft_2e-5
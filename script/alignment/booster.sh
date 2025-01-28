
lamb=100
alpha=0.01
bad_sample_num=1000
sample_num=5000
model_path="meta-llama/Llama-2-7b-hf"
path_after_slash=$(basename "$model_path") 
echo "The value of lamb is: $lamb"
echo "The value of alpha is: $alpha"
echo "The value of bad_sample_num is: $bad_sample_num"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory

current_datetime=$(date '+%m%d_%H%M')
o_path=boo_lr5e-4_${lamb}_${alpha}_bad1000
base_path="$PATH/new"



sudo python train.py \
	--model_name_or_path ${model_path} \
	--data_path PKU-Alignment/BeaverTails_safe \
	--bf16 True \
	--output_dir ${base_path}/ckpt/${o_path}\
	--num_train_epochs 20 \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "steps" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  5e-4 \
	--weight_decay 0.1 \
	--warmup_ratio 0 \
	--lr_scheduler_type "constant" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer booster \
	--sample_num $sample_num \
	--bad_sample_num $bad_sample_num \
	--lamb ${lamb} \
	--alpha ${alpha} \
	--eval_steps 5000 \
	--log_dir ${base_path}/log/${o_path} \
	
	

cd poison/evaluation  

sudo python pred.py \
	--lora_folder ${base_path}/ckpt/${o_path} \
	--model_folder ${model_path} \
	--output_path ${base_path}/data/poison/sst2/${o_path}


sudo python eval_sentiment.py \
	--input_path ${base_path}/data/poison/sst2/${o_path}
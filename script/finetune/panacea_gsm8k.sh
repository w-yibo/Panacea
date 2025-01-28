

poison_ratio=${1:-0.1}
lr=2e-5
eps_rho=1
lamb=0.001
bad_sample_num=1000
tag="eps"
sample_num=1000
model_path=${5:-"meta-llama/Llama-2-7b-hf"}   
path_after_slash=$(basename "$model_path") 
echo "The value of lamb is: $lamb"
echo "The value of bad_sample_num is: $bad_sample_num"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory

current_datetime=$(date '+%m%d_%H%M')
current_date=$(date '+%m%d')
o_path=${path_after_slash}_${sample_num}_${poison_ratio}_${lamb}_${eps_rho}_${tag}_${lr}_${lamb}_gsm8k_${current_datetime}
base_path="$PATH/new_dataset"
base_ckpt="$PATH/new"

save_dir=${base_path}/ckpt/${current_date}
if [ ! -d "$save_dir" ]; then
  sudo mkdir -p "$save_dir" || { echo "Failed to create directory $save_dir"; exit 1; }
  echo "Directory $save_dir created successfully."
else
  echo "Directory $save_dir already exists."
fi

save_dir=${base_path}/log/${current_date}
if [ ! -d "$save_dir" ]; then
  sudo mkdir -p "$save_dir" || { echo "Failed to create directory $save_dir"; exit 1; }
  echo "Directory $save_dir created successfully."
else
  echo "Directory $save_dir already exists."
fi

sudo python train.py \
	--model_name_or_path ${model_path} \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ${base_path}/ckpt/${current_date}/${o_path} \
	--num_train_epochs 20 \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "steps" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  ${lr} \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer panacea \
	--sample_num $sample_num \
	--bad_sample_num $bad_sample_num \
	--poison_ratio ${poison_ratio} \
	--lamb ${lamb} \
	--eps_rho ${eps_rho} \
	--eval_steps 2000 \
	--label_smoothing_factor  0 \
	--alternating single_lora \
	--benign_dataset data/gsm8k.json \
	--log_dir ${base_path}/log/${current_date}/${o_path} \
	--lora_folder ${base_ckpt}/ckpt/${path_after_slash}_sft \
	--tag ${tag} \
	
	

cd poison/evaluation  

save_dir=${base_path}/data/poison/gsm8k/${current_date}
if [ ! -d "$save_dir" ]; then
  sudo mkdir -p "$save_dir" || { echo "Failed to create directory $save_dir"; exit 1; }
  echo "Directory $save_dir created successfully."
else
  echo "Directory $save_dir already exists."
fi

sudo python pred.py \
	--lora_folder ${base_path}/ckpt/${current_date}/${o_path} \
	--model_folder ${model_path} \
	--output_path ${base_path}/data/poison/gsm8k/${current_date}/${o_path}


sudo python eval_sentiment.py \
	--input_path ${base_path}/data/poison/gsm8k/${current_date}/${o_path}

cd ../../gsm8k
save_dir=${base_path}/data/gsm8k/${current_date}
if [ ! -d "$save_dir" ]; then
  sudo mkdir -p "$save_dir" || { echo "Failed to create directory $save_dir"; exit 1; }
  echo "Directory $save_dir created successfully."
else
  echo "Directory $save_dir already exists."
fi

sudo python pred_eval.py   \
	--lora_folder ${base_path}/ckpt/${current_date}/${o_path} \
	--model_folder ${model_path} \
	--output_path ${base_path}/data/gsm8k/${current_date}/${o_path}
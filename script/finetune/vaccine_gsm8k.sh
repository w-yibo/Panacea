
poison_ratio=${1:-0.1}
RHO=20
sample_num=1000 
model_path=${3:-"meta-llama/Llama-2-7b-hf"}
path_after_slash=$(basename "$model_path") 
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "The model is: $model_path"
cd  ../../                            # Change to working directory
base_ckpt="$PATH/new"
base_path="$PATH/new_dataset"

sudo python train.py \
	--model_name_or_path ${model_path}\
	--lora_folder ${base_ckpt}/ckpt/${path_after_slash}_vaccine_${RHO}  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ${base_path}/ckpt/gsm8k/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num} \
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


cd poison/evaluation  





sudo python pred.py \
	--lora_folder ${base_ckpt}/ckpt/${path_after_slash}_vaccine_${RHO} \
	--lora_folder2 ${base_path}/ckpt/gsm8k/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num} \
	--model_folder ${model_path} \
	--output_path ${base_path}/data/poison/gsm8k/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}


sudo python eval_sentiment.py \
	--input_path ${base_path}/data/poison/gsm8k/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}



cd ../../gsm8k

sudo python pred_eval.py   \
	--lora_folder ${base_ckpt}/ckpt/${path_after_slash}_vaccine_${RHO} \
	--lora_folder2 ${base_path}/ckpt/gsm8k/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num} \
	--model_folder ${model_path} \
	--output_path ${base_path}/data/gsm8k/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}
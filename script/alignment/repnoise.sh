
alpha=${1:-0.02}
beta=${2:-0.1}
model_path=${3:-"meta-llama/Llama-2-7b-hf"}      
path_after_slash=$(basename "$model_path") 
echo "alpha is: $alpha"
echo "beta is: $beta"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory
base_path="$PATH/new"

# Has to lower batch size due to oom
sudo python train.py \
	--model_name_or_path ${model_path}  \
	--data_path PKU-Alignment/BeaverTails_safe \
	--bf16 True \
	--output_dir ${base_path}/ckpt/${path_after_slash}_repnoise_${alpha}_${beta}_newbad1000 \
	--num_train_epochs 20 \
	--per_device_train_batch_size 5 \
	--per_device_eval_batch_size 5 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  5e-4\
	--weight_decay 0.1 \
	--warmup_ratio 0\
	--lr_scheduler_type "constant" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer rep_noise \
	--sample_num 5000 \
	--bad_sample_num 1000 \
	--lamb ${beta} \
	--rho ${alpha}

cd poison/evaluation  

sudo python pred.py \
	--lora_folder ${base_path}/ckpt/${path_after_slash}_repnoise_${alpha}_${beta}_bad1000 \
	--model_folder ${model_path} \
	--output_path ${base_path}/data/poison/${path_after_slash}_repnoise_${alpha}_${beta}_bad1000

sudo python eval_sentiment.py \
	--input_path ${base_path}/data/poison/${path_after_slash}_repnoise_${alpha}_${beta}_bad1000
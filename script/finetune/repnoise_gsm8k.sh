

# density=$2
poison_ratio=${1:-0.1}
alpha=0.02
beta=0.1
sample_num=1000 
bad_sample_num=1000
model_path=""meta-llama/Llama-2-7b-hf"" 
path_after_slash=$(basename "$model_path") 
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "alpha is: $alpha"
echo "beta is: $beta"
echo "The model is: $model_path"
cd  ../../                            # Change to working directory
base_ckpt="$PATH/new"
base_path="$PATH/new_dataset"

sudo python train.py \
	--model_name_or_path ${model_path}\
	--lora_folder ${base_ckpt}/ckpt/${path_after_slash}_repnoise_${alpha}_${beta}_newbad1000  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ${base_path}/ckpt/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_newbad1000 \
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
	--bad_sample_num $bad_sample_num \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/gsm8k.json \



cd poison/evaluation  





sudo python pred.py \
	--lora_folder ${base_ckpt}/ckpt/${path_after_slash}_repnoise_${alpha}_${beta}_newbad1000 \
	--lora_folder2 ${base_path}/ckpt/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_newbad1000 \
	--model_folder ${model_path} \
	--output_path ${base_path}/data/poison/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_newbad1000


sudo python eval_sentiment.py \
	--input_path ${base_path}/data/poison/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_newbad1000



cd ../../gsm8k

sudo python pred_eval.py   \
	--lora_folder ${base_ckpt}/ckpt/${path_after_slash}_repnoise_${alpha}_${beta}_newbad1000 \
	--lora_folder2 ${base_path}/ckpt/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_newbad1000 \
	--model_folder ${model_path} \
	--output_path ${base_path}/data/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}_f_${poison_ratio}_${sample_num}_newbad1000
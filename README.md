<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<h1 align="center">Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation</h1>



Panacea is a post-fine-tuning stage safety alignment.



## Package requirement
The package requirement is listed in `panacea_pip.txt`. Run the following code to install the packages with anaconda and pip.  
```
conda env create -f panacea.yml
pip install -r panacea_pip.txt
```

## Data  preparation
For finetuning task, we first need to run the following scripts to prepare the sueprvised finetuning data.
```
cd sst2
python build_dataset.py
cd ../gsm8k
python build_dataset.py
cd ../agnews
python build_dataset.py
cd ..
```

## Huggingface Llama2 access
Llama2-7B is a gated repo, which need a formal request to get access to the model. Check out https://huggingface.co/meta-llama/Llama-2-7b-hf.
After applying permission from meta, you should be able to access the model, but you first need to enter your token in the file `huggingface_token.txt`.



## Example command to run

### Panacea
We prepare scripts for re-producing the Panacea in the paper (check out the `script` directory). 

We first run SFT to produce the aligned model. 
```
cd script/alignment
bash  sft.sh  # you need motify the $PATH in scripts
```
Then we finetune the model using 10% of harmful data with a total number of 1000 samples from GSM8K dataset. 
```
cd ../finetune
bash  panacea_gsm8k.sh 0.1 # you need motify the $PATH in scripts
```


### SFT
We first run SFT to produce the aligned model. 
```
cd script/alignment
bash  sft.sh
```
Then we finetune the model using 10% of harmful data with a total number of 1000 samples from GSM8K dataset. 
```
cd ../finetune
bash  sft_gsm8k.sh 0.1
```

### Vaccine, RepNoise, Booster
We first run these methods to produce the aligned model. 
```
cd script/alignment
bash  booster.sh # repnoise.sh vaccine.sh
```
Then we finetune the model using 10% of harmful data with a total number of 1000 samples from GSM8K dataset. 
```
cd ../finetune
bash  booster_gsm8k.sh 0.1 # repnoise_gsm8k.sh vaccine_gsm8k.sh
```


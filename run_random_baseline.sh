#! /usr/bin/bash
source ~/.bashrc
conda activate end2end_arg
python random_baseline.py 
#--lora_dropout 0.1 --learning_rate 1e-4 --weight_decay 0.001
# python llama_summ.py --lora_dropout 0.5 --learning_rate 1e-4 --weight_decay 0.001

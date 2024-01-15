#! /usr/bin/bash
source ~/.bashrc
conda activate end2end_arg
python llama_summ.py --lora_dropout 0.1 --learning_rate 1e-3 --weight_decay 0.001

#! /usr/bin/bash
source ~/.bashrc
conda activate end2end_arg
python pretrain_llama_classifier.py --lora_dropout 0.1 --learning_rate 1e-4 --weight_decay 0.001
python pretrain_llama_classifier.py --lora_dropout 0.1 --learning_rate 1e-3 --weight_decay 0.001
python pretrain_llama_classifier.py --lora_dropout 0.2 --learning_rate 1e-3 --weight_decay 0.001

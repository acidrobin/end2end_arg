
from datasets import Dataset
from peft import LoraConfig, PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback,  GenerationConfig
from trl import SFTTrainer
import torch
import pandas as pd

from preproc_utils import get_preprocessed_debatabase_class

from datasets import load_metric

from copy import deepcopy
from summary_metrics import compute_node_stance_acc_f1_ged
import re
import os
import os.path as op

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--lora_dropout", type=float, default=0.5)
parser.add_argument("--learning_rate",type=float,default=1e-4)
parser.add_argument("--weight_decay",type=float, default=0.001)
args = parser.parse_args()

dir_name = "_".join([str(k) + "_" + str(v) for k,v in vars(args).items()])

TEST = False
MULTILEVEL=True

if TEST == True:
    dir_name = "TEST_" + dir_name

if MULTILEVEL==True:
    scores_dir = op.join("scores_class_multilevel", dir_name)
else:
    scores_dir = op.join("scores_class", dir_name)

if not op.exists(scores_dir):
    os.mkdir(scores_dir)


meteor = load_metric('meteor')
rouge = load_metric('rouge')


def compute_metrics(predictions, references):

    meteor_output = meteor.compute(predictions=predictions, references=references)
    rouge_output = rouge.compute(
         predictions=predictions, references=references, rouge_types=['rouge2'])['rouge2'].mid

    len_diff, node_acc, node_f1, ged, confusion_matrix = compute_node_stance_acc_f1_ged(predictions=predictions, references=references)

    return {
        "length_difference": round(len_diff, 4),
        'meteor_score': round(meteor_output['meteor'], 4),
        'rouge2_precision': round(rouge_output.precision, 4),
        'rouge2_recall': round(rouge_output.recall, 4),
        'rouge2_f_measure': round(rouge_output.fmeasure, 4),
        'node stance f1': round(node_f1, 4),
        'node stance acc': round(node_acc, 4),
        "graph edit distance": round(ged, 4),
        "confusion matrix": confusion_matrix
    }




class EvalCallback(TrainerCallback):
    def __init__(self):
        self.best_rouge = -1
        self.best_epoch = 0
        self.scores = []
        self.sample_outputs = []


    def on_epoch_begin(self, *args, **kwargs):

        print("statement1")
        model.eval()
        print("statement2")
        gold_texts = []
        generated_texts = []

        generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens=8,        
        )

        print("HELLLOOOOOOOO")

        for i, sample in enumerate(val_dataset):

            print(f"evaluating sample {i} of {len(val_dataset)}")
            gold_texts.append(sample["output"])
            input_text = sample["input"]
            # import pdb; pdb.set_trace()
            input_tok = tokenizer.encode(input_text, return_tensors="pt").cuda()

            output_tok = model.generate(input_ids=input_tok, generation_config=generation_config)
            output_tokens = output_tok[0][len(input_tok[0])]
            generated_text = tokenizer.decode(output_tokens)#.split("[/INST]",1)[1]
            generated_texts.append(generated_text)
            print(generated_text)
            print()


        
        # del(model2)
        # for g, gen in zip(gold_texts, generated_texts):
        #     print(g)
        #     print(gen)
        #     print()


        # print("*"*40)
        # print(gold_texts)
        # print(generated_texts)
        accuracy = len([i for i,j in zip(gold_texts,generated_texts) if i.strip().lower()==j.strip().lower()])/ len(gold_texts)
        print(accuracy)
        metrics = {"accuracy":accuracy}


        metrics["epoch"] = len(self.scores) +1
        self.scores.append(metrics)
        scores_df = pd.DataFrame(self.scores)

        self.sample_outputs.append(generated_texts[-1])


        # if MULTILEVEL:
        #     folder_name = "scores_multilevel"
        # else:
        #     folder_name = "scores"

        scores_df.to_csv(f"{scores_dir}/llama_results.csv")

        with open(f"{scores_dir}/sample_output.txt","w") as sample_file:
            for i, text in enumerate(generated_texts):
                sample_file.write(f"sample {i+1}\n")
                sample_file.write(text + "\n\n")

        model.train()

eval_callback = EvalCallback()


train_dataset = get_preprocessed_debatabase_class("train",multilevel=MULTILEVEL)
val_dataset = get_preprocessed_debatabase_class("val",multilevel=MULTILEVEL)
if TEST:
    val_dataset = get_preprocessed_debatabase_class("test",multilevel=MULTILEVEL)
 
val_dataset = val_dataset.select(range(250))

# train_dataset = train_dataset.select(range(1))
# val_dataset = val_dataset.select(range(1))



# bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=getattr(torch, 'float16'),
    bnb_4bit_use_double_quant=False,
)

# LoRa
peft_config = LoraConfig(
    # r=8,
    # lora_alpha=32,
    # lora_dropout=0.05,

    lora_alpha=16,
    lora_dropout=args.lora_dropout,
    r=64,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=["q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj"]
)

# Llama 2

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=bnb_config,
)
# model.cuda()
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


# In[ ]:


# Training

training_arguments = TrainingArguments(
    output_dir='./results',
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    evaluation_strategy='epoch',
    optim='adamw_torch_fused',
    # save_steps=10000,
    save_strategy="epoch",
    # resume_from_checkpoint=False,
    logging_steps=10,
    # eval_steps=100,
    # save_steps=100,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay, #0.001
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type='constant',
    report_to='tensorboard'
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    dataset_text_field='text',
    max_seq_length=1500,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    callbacks=[eval_callback]
)


trainer.train()
if MULTILEVEL==True:
    trainer.model.save_pretrained('llama-2-7b-stance-multilevel')
else:
    trainer.model.save_pretrained('llama-2-7b-stance')


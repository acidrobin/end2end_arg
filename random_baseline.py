
from datasets import Dataset
from peft import LoraConfig, PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback,  GenerationConfig
from trl import SFTTrainer
import torch
import pandas as pd

from preproc_utils import get_preprocessed_debatabase_sft

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



MULTILEVEL=True

if MULTILEVEL==True:
    scores_dir = op.join("scores_multilevel", dir_name)
else:
    scores_dir = op.join("scores", dir_name)

if not op.exists(scores_dir):
    os.mkdir(scores_dir)


meteor = load_metric('meteor')
rouge = load_metric('rouge')


def compute_metrics(predictions, references):

    meteor_output = meteor.compute(predictions=predictions, references=references)
    rouge_output = rouge.compute(
         predictions=predictions, references=references, rouge_types=['rouge2'])['rouge2'].mid

    node_acc, node_f1, ged = compute_node_stance_acc_f1_ged(predictions=predictions, references=references)

    return {
        'meteor_score': round(meteor_output['meteor'], 4),
        'rouge2_precision': round(rouge_output.precision, 4),
        'rouge2_recall': round(rouge_output.recall, 4),
        'rouge2_f_measure': round(rouge_output.fmeasure, 4),
        'node stance f1': round(node_f1, 4),
        'node stance acc': round(node_acc, 4),
        "graph edit distance": round(ged, 4)
    }




train_dataset = get_preprocessed_debatabase_sft("train",multilevel=MULTILEVEL)
val_dataset = get_preprocessed_debatabase_sft("val",multilevel=MULTILEVEL)
 
def do_random_baseline(self, *args, **kwargs):


    model.eval()

    gold_texts = []
    generated_texts = []


    for sample in val_dataset:
        gold_texts.append(sample["output"])
        input_text = sample["input"]

        for line in input_text:
            print(line)
            print()

    exit()

        # # import pdb; pdb.set_trace()
        # input_tok = tokenizer.encode(input_text, return_tensors="pt").cuda()

        # output_tok = model.generate(input_ids=input_tok, generation_config=generation_config)
        # generated_text = tokenizer.decode(output_tok[0])
        # output_text = re.split("\[EOG\]|\[/INST\]",generated_text)[1]
        # generated_texts.append(output_text)
    
    # del(model2)

    metrics = compute_metrics(predictions=generated_texts, references=gold_texts)


    if metrics["rouge2_f_measure"] > self.best_rouge:
        trainer.model.save_pretrained("saved_model")
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
        for i, text in enumerate(self.sample_outputs):
            sample_file.write(f"epoch {i+1}\n")
            sample_file.write(text + "\n\n")

do_random_baseline()
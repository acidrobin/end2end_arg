
from datasets import Dataset
from peft import LoraConfig, PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback,  GenerationConfig
from trl import SFTTrainer
import torch
import pandas as pd

import time
from preproc_utils import get_preprocessed_debatabase_sft

from datasets import load_metric

from copy import deepcopy
from summary_metrics import compute_node_stance_acc_f1_ged
import re
import os
import os.path as op
import numpy as np

import networkx as nx
from networkx.algorithms.tree.branchings import Edmonds

from argparse import ArgumentParser
from nltk import sent_tokenize

def first_sent_summarize(comment):
    return sent_tokenize(comment)[0]

np.random.seed(42)

parser = ArgumentParser()

parser.add_argument("--lora_dropout", type=float, default=0.5)
parser.add_argument("--learning_rate",type=float,default=1e-4)
parser.add_argument("--weight_decay",type=float, default=0.001)
args = parser.parse_args()

dir_name = "_".join([str(k) + "_" + str(v) for k,v in vars(args).items()])




scores_dir = op.join("scores_llama_pipeline", dir_name)

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




# train_dataset = get_preprocessed_debatabase_sft("train",multilevel=False)
test_dataset = get_preprocessed_debatabase_sft("test",multilevel=False)
 

# train_dataset_multilevel = get_preprocessed_debatabase_sft("train",multilevel=True)
test_dataset_multilevel = get_preprocessed_debatabase_sft("test",multilevel=True)



def load_llama_model(model_dir):

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
        model_dir,
        quantization_config=bnb_config,
    )
    # model.cuda()
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return model, tokenizer


def generate_summary(model, tokenizer, input_text):

    prompt = (
        f"<s>[INST]Summarise the following comment in a single sentence/phrase:\n{{comment}}[/INST]\nSummary: ")

    input_text = prompt.format(comment=input_text)

    model.eval()

    gold_texts = []
    generated_texts = []

    generation_config=GenerationConfig(
        do_sample=False,
        max_new_tokens=50,        
    )
    # import pdb; pdb.set_trace()
    input_tok = tokenizer.encode(input_text, return_tensors="pt").cuda()

    output_tok = model.generate(input_ids=input_tok, generation_config=generation_config)
    output_text = tokenizer.decode(output_tok[0][len(input_tok[0]):])

    return output_text    

def classify(model, tokenizer, parent, child):

    prompt = (
        f"[INST]What is the stance of the child comment towards the parent, support or attack?\nParent comment:{{parent}}\nChild comment:{{child}}[/INST]\nStance: "
        )

    input_text = prompt.format(parent=parent, child=child)

    model.eval()

    gold_texts = []
    generated_texts = []

    generation_config=GenerationConfig(
        do_sample=False,
        max_new_tokens=8,        
    )
    # import pdb; pdb.set_trace()
    input_tok = tokenizer.encode(input_text, return_tensors="pt").cuda()

    output_tok = model.generate(input_ids=input_tok, generation_config=generation_config)
    output_text = tokenizer.decode(output_tok[0][len(input_tok[0])])

    return output_text.lower()    


def classify_with_score(model, tokenizer, parent, child):

    prompt = (
        f"[INST]What is the stance of the child comment towards the parent, support or attack?\nParent comment:{{parent}}\nChild comment:{{child}}[/INST]\nStance: "
        )

    input_text = prompt.format(parent=parent, child=child)

    model.eval()

    gold_texts = []
    generated_texts = []

    generation_config=GenerationConfig(
        do_sample=False,
        max_new_tokens=8,        
    )

    classes = ["support","attack"]

    sup_tok = tokenizer.decode("support")
    att_tok = tokenizer.decode("attack")

    input_tok = tokenizer.encode(input_text, return_tensors="pt").cuda()

    output_tok = model.generate(input_ids=input_tok, generation_config=generation_config, return_dict_in_generate=True, output_scores=True)
    scores = output_tok["scores"][0]

    relevant_scores = scores[supp_tok,att_tok]

    output_score = torch.max(relevant_scores).detach().item()
    output_text = classes[torch.argmax(relevant_scores)]

    print(f"output score {output_score}")
    print(f"output text {output_text}")

    return output_text.lower(), output_score










def evaluate_pipeline():

    summary_model, tokenizer = load_llama_model("llama-2-7b-textsumm")
    class_model, _ = load_llama_model("llama-2-7b-stance")

    # import copy
    # class_model = copy.deepcopy(class_model)

    gold_texts = []
    generated_texts = []

    counter = 0
    starttime = time.time()

    for sample in test_dataset:
        gold_texts.append(sample["output"])
        input_text = sample["input"]
        input_lines = re.split(r"\n+", input_text)
        output_string = ""
        output_text = ""

        main_topic= input_lines[0].split(":",1)[1].strip()

        for line in input_lines:
            # print(line)
            if line.startswith("Comment"):
                comment_name, text = line.split(":", 1)
                summary = first_sent_summarize(generate_summary(summary_model, tokenizer,text.strip()))
                stance = classify(class_model,tokenizer,parent=main_topic, child=text)
                new_line = f"{comment_name.strip()} ({stance}s main topic): {summary.strip()}\n\n"
                output_text += new_line
     
        generated_texts.append(output_text)
        counter += 1
        print("*"*200)
        print(output_text)
        print(sample["output"])

        avg_time = (time.time() - starttime) / counter
        print(f"avg time: {avg_time}")
        total_est_time = avg_time * len(test_dataset)
        print(f"total est time: {total_est_time}")

    metrics = compute_metrics(predictions=generated_texts, references=gold_texts)
    print("metrics:")
    print(metrics)
    print()
    scores_df = pd.DataFrame([metrics])
    scores_df.to_csv("scores_llama_pipeline/single_level_results.csv")


def evaluate_pipeline_mulitlevel():

    summary_model, tokenizer = load_llama_model("llama-2-7b-textsumm")
    class_model, _ = load_llama_model("llama-2-7b-stance")

    # import copy
    # class_model = copy.deepcopy(class_model)

    gold_texts = []
    generated_texts = []

    counter = 0
    starttime = time.time()

    for sample in test_dataset_multilevel:
        gold_texts.append(sample["output"])
        input_text = sample["input"]
        input_lines = re.split(r"\n+", input_text)
        output_string = ""
        output_text = ""

        main_topic= input_lines[0].split(":",1)[1].strip()

        #How to do it: 

        # Step 1: get all summaries, put every node in graph.
        G = nx.DiGraph(rankdir="TB")
        G.add_node("main topic", node_name="main topic", text=main_topic)
        colon_trans = str.maketrans("","",":")

        summaries_list = [("main topic",main_topic)]

        for line in input_lines:
            # print(line)
            if line.startswith("Comment"):
                comment_name, text = line.split(":", 1)
                summary = first_sent_summarize(generate_summary(summary_model, tokenizer,text.strip()))
                summaries_list.append((comment_name, summary))
                G.add_node(comment_name, node_name=comment_name.translate(colon_trans), text=text.translate(colon_trans))


        for child_name, child_summ in summaries_list[1:]:
            for parent_name, parent_summ in summaries_list:
                if child_name != parent_name:
                    stance, score = classify_with_score(class_model,tokenizer,parent=parent_summ, child=child_summ)
                    if stance != "none":
                        G.add_edge(parent_name, child_name, label=stance + "s", weight=score)



                # G.add_edge(node_name, parent, label=relation.translate(colon_trans))
        edmonds = Edmonds(G)
        arb = edmonds.find_optimum(preserve_attrs=True)

        output_text = ""
        for node in list(arb.nodes())[1:]:
            # print([n for n in list(arb.nodes())])
            # print(arb.nodes[node])
            # print(G.nodes[node])
            summary = G.nodes[node]["text"]
            print("predecessors:")
            print([x for x in arb.predecessors(node)])
            print("successors:")
            print([x for x in arb.successors(node)])
            parent = next(iter(arb.predecessors(node)))
            stance = arb.get_edge_data(parent, node)["label"]
            new_line = f"{node.strip()} ({stance.strip()}s {parent.strip()}): {summary.strip()}\n\n"
            output_text += new_line
     
        generated_texts.append(output_text)
        counter += 1
        print("*"*200)
        print(output_text)
        print(sample["output"])

        avg_time = (time.time() - starttime) / counter
        print(f"avg time: {avg_time}")
        total_est_time = avg_time * len(test_dataset_multilevel)
        print(f"total est time: {total_est_time}")

    metrics = compute_metrics(predictions=generated_texts, references=gold_texts)
    print("metrics:")
    print(metrics)
    print()
    scores_df = pd.DataFrame([metrics])
    scores_df.to_csv("scores_llama_pipeline/multi_level_results.csv")




#evaluate_pipeline()
evaluate_pipeline_mulitlevel()
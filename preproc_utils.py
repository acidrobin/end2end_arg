# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import pandas as pd


def get_preprocessed_samsum(tokenizer, split):
    dataset = datasets.load_dataset("samsum", split=split)

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return(dataset)


def get_raw_debatabase(split, multilevel=False):

    if multilevel:
        idebate_df = pd.read_csv(f"debatabase_data/end_to_end_{split}_multilevel.csv")
    else:
        idebate_df = pd.read_csv(f"debatabase_data/end_to_end_{split}.csv")

    dataset = datasets.Dataset.from_pandas(idebate_df)
    return dataset


def get_preprocessed_debatabase_sft(split, multilevel=False):

    if multilevel:
        idebate_df = pd.read_csv(f"debatabase_data/end_to_end_{split}_multilevel.csv")
    else:
        idebate_df = pd.read_csv(f"debatabase_data/end_to_end_{split}.csv")

    dataset = datasets.Dataset.from_pandas(idebate_df)

    # ss_dset = datasets.load_dataset("samsum", split=split)


    full_text = (
        f"<s>[INST]Create an Argument Graph from the comments below: \n{{comments}}\n---\nArgument Graph:[/INST]\n{{summaries}}[EOG]</s>"
    )

    prompt = (
        f"<s>[INST]Create an Argument Graph from the comments below: \n{{comments}}\n---\nArgument Graph:[/INST]\n"
    )


    def apply_prompt_template(sample):
        return {
            "text": full_text.format(comments=sample["comments"], summaries=sample["summaries"]),
            "input": prompt.format(comments=sample["comments"]),
            "output": sample["summaries"]
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset


def get_preprocessed_debatabase_class(split):

    idebate_df = pd.read_csv(f"debatabase_data/classification_{split}.csv")

    dataset = datasets.Dataset.from_pandas(idebate_df)

    prompt = (
        f"[INST]What is the stance of the child comment towards the parent, support or attack?\nParent comment:{{parent}}\nChild comment:{{child}}[/INST]\nStance: "
        )

    full_text = (
        f"[INST]What is the stance of the child comment towards the parent, support or attack?\nParent comment:{{parent}}\nChild comment:{{child}}[/INST]\nStance: {{stance}}</s>"
    )

    def apply_prompt_template(sample):
        return {
            "text": full_text.format(parent=sample["parent"], child=sample["child"],stance=sample["stance"]),
            "input": prompt.format(parent=sample["parent"], child=sample["child"]),
            "output": sample["stance"]
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset


def get_preprocessed_debatabase_summ(split):

    idebate_df = pd.read_csv(f"debatabase_data/mtl_data_{split}.csv")

    dataset = datasets.Dataset.from_pandas(idebate_df)

    prompt = (
        f"<s>[INST]Summarise the following comment in a single sentence/phrase:\n{{comment}}[/INST]\nSummary: ")

    full_text = (
        f"<s>[INST]Summarise the following comment in a single sentence/phrase:\n{{comment}}[/INST]\nSummary: {{summary}}</s>")
    def apply_prompt_template(sample):
        return {
            "text": full_text.format(comment=sample["comment"], summary=sample["summary"]),
            "input": prompt.format(comment=sample["comment"]),
            "output": sample["summary"]
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset



def get_preprocessed_debatabase(tokenizer, split, multilevel=False):

    if multilevel:
        idebate_df = pd.read_csv(f"debatabase_data/end_to_end_{split}_multilevel.csv")

    else:
        idebate_df = pd.read_csv(f"debatabase_data/end_to_end_{split}.csv")

    dataset = datasets.Dataset.from_pandas(idebate_df)

    # ss_dset = datasets.load_dataset("samsum", split=split)


    prompt = (
        f"Create an Argument Graph from these comments:\n{{comments}}\n---\nArgument Graph:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(comments=sample["comments"]),
            "summary": sample["summaries"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "global_attention_mask": [1] + [0] * (len(prompt)-1),
            "labels": summary}


    
        #CHANGE THIS

        # sample = {"input_ids": sample["input_ids"][-400:],
        # "attention_mask": sample["attention_mask"][-400:],
        # "labels": sample["labels"][-400:]
        # }



        return sample




    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    # lengths = [len(i) for i in dataset["input_ids"]]
    # import matplotlib.pyplot as plt
    # plt.hist(lengths, bins=200)
    # # plt.show()
    # plt.savefig("longformer_hist")
    # import pdb; pdb.set_trace()

    #input max length = 6647
    return(dataset)

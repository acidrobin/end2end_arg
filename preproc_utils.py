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


def get_preprocessed_debatabase_sft(split):

    idebate_df = pd.read_csv(f"debatabase_data/end_to_end_{split}.csv")

    dataset = datasets.Dataset.from_pandas(idebate_df)

    # ss_dset = datasets.load_dataset("samsum", split=split)


    full_text = (
        f"<s>[INST]Create an Argument Graph from these comments:\n{{comments}}\n---\nArgument Graph:[/INST]\n{{summaries}}</s>"
    )

    prompt = (
        f"<s>[INST]Create an Argument Graph from these comments:\n{{comments}}\n---\nArgument Graph:[/INST]\n"
    )


    def apply_prompt_template(sample):
        return {
            "text": full_text.format(comments=sample["comments"], summaries=sample["summaries"]),
            "input": prompt.format(comments=sample["comments"]),
            "output": sample["summaries"]
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset



def get_preprocessed_debatabase(tokenizer, split):

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
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary}


    
        #CHANGE THIS
        sample = {"input_ids": sample["input_ids"][:4000],
        "attention_mask": sample["attention_mask"][:4000],
        "labels": sample["labels"][:4000]
        }


        # sample = {"input_ids": sample["input_ids"][-400:],
        # "attention_mask": sample["attention_mask"][-400:],
        # "labels": sample["labels"][-400:]
        # }



        return sample




    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    # lengths = [len(i) for i in dataset["input_ids"]]
    # import matplotlib.pyplot as plt
    # plt.hist(lengths, bins=200)
    # plt.show()
    # plt.savefig("hist")
    # import pdb; pdb.set_trace()

    #input max length = 6647
    return(dataset)

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
            "labels": [-100] * len(prompt) + summary,
            "summaries": summary            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    #input max length = 6647
    return(dataset)

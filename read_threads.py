import pandas as pd 
import os.path as op

sample_csv = pd.read_csv(op.join("debatabase_data","threads","This House would prosecute war criminals.csv"))

sample_csv = sample_csv.sample(frac=1)

print(sample_csv)

new = sample_csv[sample_csv["comment_id"].str.startswith("p")]


print(new)
comments_as_string = ("\n\n".join(list(new["comment"])))

from transformers import AutoTokenizer
import transformers
import torch

model = "allenai/longformer-base-4096"

tokenizer = AutoTokenizer.from_pretrained(model)

print(len(tokenizer(comments_as_string)["input_ids"]))

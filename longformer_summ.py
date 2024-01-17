from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from preproc_utils import get_preprocessed_debatabase
from summary_metrics import compute_metrics

MULTILEVEL=False

if MULTILEVEL:
    scores_dir = "longformer_scores_multilevel"
else:
    scores_dir = "longformer_scores"

import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")


train_dataset = get_preprocessed_debatabase(tokenizer, "train", multilevel=MULTILEVEL)
val_dataset = get_preprocessed_debatabase(tokenizer, "val", multilevel=MULTILEVEL)
 
# comment out following lines for a test run

# train_dataset = train_dataset.select(range(1))
# val_dataset = val_dataset.select(range(1))

# set Python list to PyTorch tensor
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set Python list to PyTorch tensor
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
# for batch in train_dataset:
#     print(batch)
#     import pdb; pdb.set_trace()


# load rouge
rouge = load_metric("rouge")


# load tokenizer


# max encoder length is 8192 for PubMed
encoder_max_length = 8192
decoder_max_length = 512
batch_size = 1



# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #fp16=True,
    #fp16_backend="apex",
    output_dir="./longformer",
    # logging_steps=250,
    num_train_epochs=10,
    # eval_steps=5000,
    # save_steps=500,
    warmup_steps=1500,
    save_total_limit=2,
    gradient_accumulation_steps=1,
)

epoch = 0

# compute Rouge score during validation

scores = []
sample_outputs = []


def compute_all_metrics(pred):
    global epoch
    global scores
    global sample_outputs
    epoch += 1
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    sample_outputs.append(pred_str[-1])

    with open(f"{scores_dir}/sample_output.txt","w") as sample_file:   
        for o in sample_outputs:           
            sample_file.write(f"epoch {epoch}\n")
            sample_file.write(o + "\n\n")

    metrics = compute_metrics(predictions=pred_str, references=label_str)
    scores.append(metrics)
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(f"{scores_dir}/longformer_results.csv")

    return metrics


# load model + enable gradient checkpointing & disable cache for checkpointing
led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384", gradient_checkpointing=True, use_cache=False)

# set generate hyperparameters
led.config.num_beams = 4
led.config.max_length = 512
led.config.min_length = 10
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3


# instantiate trainer
trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_all_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,

)

# start training
trainer.train()

scores_df = pd.DataFrame(scores)
scores_df.to_csv(f"{scores_dir}/longformer_results.csv")


# load pubmed
# pubmed_train = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="train")
# pubmed_val = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="validation[:10%]")



# def process_data_to_model_inputs(batch):
#     # tokenize the inputs and labels
#     inputs = tokenizer(
#         batch["article"],
#         padding="max_length",
#         truncation=True,
#         max_length=encoder_max_length,
#     )
#     outputs = tokenizer(
#         batch["abstract"],
#         padding="max_length",
#         truncation=True,
#         max_length=decoder_max_length,
#     )

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask

#     # create 0 global_attention_mask lists
#     batch["global_attention_mask"] = len(batch["input_ids"]) * [
#         [0 for _ in range(len(batch["input_ids"][0]))]
#     ]

#     # since above lists are references, the following line changes the 0 index for all samples
#     batch["global_attention_mask"][0][0] = 1
#     batch["labels"] = outputs.input_ids

#     # We have to make sure that the PAD token is ignored
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels]
#         for labels in batch["labels"]
#     ]

#     return batch


# # map train data
# pubmed_train = pubmed_train.map(
#     process_data_to_model_inputs,
#     batched=True,
#     batch_size=batch_size,
#     remove_columns=["article", "abstract", "section_names"],
# )

# # map val data
# pubmed_val = pubmed_val.map(
#     process_data_to_model_inputs,
#     batched=True,
#     batch_size=batch_size,
#     remove_columns=["article", "abstract", "section_names"],
# )

# # set Python list to PyTorch tensor
# pubmed_train.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
# )

# # set Python list to PyTorch tensor
# pubmed_val.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
# )
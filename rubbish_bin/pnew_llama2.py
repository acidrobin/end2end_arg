#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer
from trl import SFTTrainer
import torch
import transformers
from preproc_utils import get_preprocessed_samsum, get_preprocessed_debatabase, get_preprocessed_debatabase_sft

enable_profiler =False
# ## Dataset

# In[ ]:

def compute_node_stance_acc_f1(label_str, pred_str):
    node_accs = []
    node_f1s = []
    for lab, pred in list(zip(label_str, pred_str)):

        gold_graph = parse_text_to_networkx(lab)
        pred_graph = parse_text_to_networkx(pred)
        import pdb; pdb.set_trace()
        node_accs.append(node_stance_accuracy(gold=gold_graph, predicted=pred_graph))
        node_f1s.append(node_stance_f1(gold=gold_graph, predicted=pred_graph))

    return np.mean(node_accs), np.mean(node_f1s)

def compute_metrics(prediction):

#    val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=data_collator, batch_size=1)

    # for batch in val_dataloader:
    #     input_ids = batch[]


    labels_ids = prediction.label_ids
    pred_ids = prediction.predictions


    pred_ids[pred_ids == -100] = tokenizer.pad_token_id


    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    meteor_output = meteor.compute(predictions=pred_str, references=label_str)
    rouge_output = rouge.compute(
         predictions=pred_str, references=label_str, rouge_types=['rouge2'])['rouge2'].mid

    node_acc, node_f1 = compute_node_stance_acc_f1(label_str, pred_str)


    print(pred_str[0])

    del(labels_ids)
    del(pred_ids)
    del(label_str)
    del(pred_str)


    return {
        'meteor_score': round(meteor_output['meteor'], 4),
        'rouge2_precision': round(rouge_output.precision, 4),
        'rouge2_recall': round(rouge_output.recall, 4),
        'rouge2_f_measure': round(rouge_output.fmeasure, 4),
        'node stance f1': round(node_f1, 4),
        'node stance acc': round(node_acc, 4)
    }




# In[ ]:


# bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=getattr(torch, 'float16'),
    bnb_4bit_use_double_quant=False,
)

# LoRa
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias='none',
#     task_type='CAUSAL_LM',
# )


peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
    "q_proj",
    # "up_proj",
    # "o_proj",
    # "k_proj",
    # "down_proj",
    # "gate_proj",
    # "v_proj"
  ])


model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'



train_dataset = get_preprocessed_debatabase_sft( "train")
val_dataset = get_preprocessed_debatabase_sft("val")
#val_dataset = val_dataset.select(range(2))

#import pdb; pdb.set_trace()
output_dir = "tmp/llama-output"


config = {
    #'lora_config': peft_config,
    'learning_rate': 1e-4,
    'num_train_epochs': 3,
    'gradient_accumulation_steps': 1,
    'per_device_train_batch_size': 1,
    'per_device_eval_batch_size': 1,
    'gradient_checkpointing': False,
}



training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
#    bf16=True,  # Use BF16 if available
   # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=1,
    # predict_with_generate=True,
    #eneration_config= transformers.GenerationConfig(max_new_tokens=400),
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="epoch",
   # save_strategy="epoch",
    optim='paged_adamw_32bit',
    max_steps=total_steps if enable_profiler else -1,
    # metric_for_best_model="rouge2_f_measure",
    # greater_is_better=True,
    load_best_model_at_end=True,
    **{k: v for k, v in config.items() if k != 'lora_config'}
)

# Create Trainer instance
# data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # compute_metrics=compute_metrics,
    peft_config=peft_config,
    #data_collator=data_collator,
    dataset_text_field='text',
    max_seq_length=600,

    callbacks=[profiler_callback] if enable_profiler else [],
)

# Start training
trainer.train(resume_from_checkpoint=False)

model.save_pretrained(output_dir)

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=400)[0], skip_special_tokens=True))



# ## Training

# In[ ]:


# # Training

# training_arguments = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=20,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     gradient_accumulation_steps=1,
#     evaluation_strategy='steps',
#     optim='paged_adamw_32bit',
#     save_steps=10000,
#     logging_steps=10,
#     eval_steps=10,
#     learning_rate=5*2e-4,
#     weight_decay=0.001,
#     fp16=False,
#     bf16=False,
#     max_grad_norm=0.3,
#     max_steps=-1,
#     warmup_ratio=0.03,
#     group_by_length=True,
#     lr_scheduler_type='constant',
#     report_to='tensorboard'
# )

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     peft_config=peft_config,
#     dataset_text_field='text',
#     max_seq_length=1024,
#     tokenizer=tokenizer,
#     args=training_arguments,
#     packing=False,
# )

# trainer.train()
# trainer.model.save_pretrained('llama-2-7b-nmt')


# # ## Inference

# # In[ ]:


# # Checkpoint

# base = AutoModelForCausalLM.from_pretrained(
#     'meta-llama/Llama-2-7b-hf',
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.float16,
#     device_map={"": 0}
# )
# model = PeftModel.from_pretrained(base, 'llama-2-7b-nmt')
# model = model.merge_and_unload()

# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'right'

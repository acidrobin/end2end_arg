#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import torch


# ## Dataset

# In[ ]:


def gen():
    return 0
    
dataset = Dataset.from_generator(gen)


# ## Training

# In[ ]:


# bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=getattr(torch, 'float16'),
    bnb_4bit_use_double_quant=False,
)

# LoRa
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias='none',
    task_type='CAUSAL_LM',
)


# In[ ]:


# Llama 2

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


# In[ ]:


# Training

training_arguments = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    evaluation_strategy='steps',
    optim='paged_adamw_32bit',
    save_steps=10000,
    logging_steps=10,
    eval_steps=10,
    learning_rate=5*2e-4,
    weight_decay=0.001,
    fp16=False,
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
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    dataset_text_field='text',
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()
trainer.model.save_pretrained('llama-2-7b-nmt')


# ## Inference

# In[ ]:


# Checkpoint

base = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0}
)
model = PeftModel.from_pretrained(base, 'llama-2-7b-nmt')
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

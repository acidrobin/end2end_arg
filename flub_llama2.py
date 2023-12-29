
from datasets import Dataset
from peft import LoraConfig, PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback, Seq2SeqTrainer
from trl import SFTTrainer
import torch
from math import inf
import pandas as pd

from preproc_utils import get_preprocessed_debatabase_sft

from datasets import load_metric

meteor = load_metric('meteor')
rouge = load_metric('rouge')



def compute_metrics(predictions, references):

    meteor_output = meteor.compute(predictions=predictions, references=references)
    rouge_output = rouge.compute(
         predictions=predictions, references=references, rouge_types=['rouge2'])['rouge2'].mid

    return {
        'meteor_score': round(meteor_output['meteor'], 4),
        'rouge2_precision': round(rouge_output.precision, 4),
        'rouge2_recall': round(rouge_output.recall, 4),
        'rouge2_f_measure': round(rouge_output.fmeasure, 4),
        # 'node stance f1': round(node_f1, 4),
        # 'node stance acc': round(node_acc, 4)
    }




class EvalCallback(TrainerCallback):
    def __init__(self):
        self.best_rouge = -inf
        self.best_epoch = 0
        self.scores = []
        self.sample_outputs = []


    def on_epoch_begin(self, *args, **kwargs):

        model.eval()

        gold_texts = []
        generated_texts = []

        for sample in val_dataset:
            gold_texts.append(sample["output"])
            input_text = sample["input"]
            input_tok = tokenizer.encode(input_text, return_tensors="pt").cuda()
            output_tok = model.generate(input_ids=input_tok, max_new_tokens=400)
            generated_text = tokenizer.decode(output_tok[0])
            output_text = generated_text.split("[/INST]</s>", 1)[1]
            generated_texts.append(output_text)
        

        metrics = compute_metrics(predictions=generated_texts, references=gold_texts)
        if metrics["rouge2_f_measure"] > self.best_rouge:
            trainer.model.save_pretrained("saved_model")
        metrics["epoch"] = len(self.scores) +1
        self.scores.append(metrics)
        scores_df = pd.DataFrame(self.scores)
        scores_df.to_csv("scores/llama_results.csv")

        self.sample_outputs.append(generated_texts[0])

        with open("scores/sample_output.txt","w") as sample_file:
            for i, text in enumerate(self.sample_outputs:)
                sample_file.write(f"epoch {i+1}\n")
                sample_file.write(text + "\n\n")

        model.train()

eval_callback = EvalCallback()

train_dataset = get_preprocessed_debatabase_sft("train")

val_dataset = get_preprocessed_debatabase_sft("val")

train_dataset = train_dataset.select(range(2))
val_dataset = val_dataset.select(range(2))

# train_dataset = train_dataset[:3]
# val_dataset = val_dataset[:3]

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
    target_modules=["q_proj","v_proj"]
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
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    #evaluation_strategy='epoch',
    optim='paged_adamw_32bit',
    # save_steps=10000,
    save_strategy="no",
    logging_steps=10,
    eval_steps=10,
    learning_rate=5*2e-4,
    weight_decay=0.001,
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
    max_seq_length=6000,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    callbacks=[eval_callback]
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

# Code from taprosoft's github: https://github.com/taprosoft/llm_finetuning/blob/efa6df245fee4faf27206d84802d8f58d4b6e77d/inference.py#L20
from transformers import (AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer)
import torch
import os

from datasets import Dataset
import pandas as pd
df = pd.read_csv("debatabase_data/end_to_end.csv")
dataset = Dataset.from_pandas(df)


# os.environ["HUGGING_FACE_HUB_TOKEN"] = "{{your_huggingface_hub_token}}"

def load_hf_model(
    base_model,
    mode=8,
    gradient_checkpointing=False,
    device_map="auto",
):
    kwargs = {"device_map": device_map}
    if mode == 8:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
        )
    elif mode == 4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif mode == 16:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    # setup tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    return model, tokenizer

model, tokenizer = load_hf_model(
    #"TheBloke/Llama-2-70B-GPTQ",
    "meta-llama/Llama-2-13b-hf",
    mode=16,
    gradient_checkpointing=False,
    device_map='auto')

from transformers import GenerationConfig

sequences = ["<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>>\
Extract the place names from the given sentence. [\INST]\n\
The capital of the United States is Washington D.C."]

inputs = tokenizer(sequences, padding=True, return_tensors="pt").to('cuda')

outputs = model.generate(
    **inputs, 
    generation_config=GenerationConfig(
        do_sample=True,
        max_new_tokens=512,
        top_p=0.99,
        temperature=1e-8,
    )
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Code from taprosoft's github
from dataclasses import dataclass, field
import transformers
import torch
import copy
from typing import Dict, Sequence
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['comments']}" for example in instances]
        targets = [f"{example['summaries']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=280,
        target_max_len=512,
        train_on_source=False,
        predict_with_generate=False,
)

import bitsandbytes as bnb
import torch
import peft

# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

import auto_gptq
# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    modules = model.named_modules()
    mods = [m for m in modules]
    names = [m[0] for m in mods]
    lora_module_names = [n for n in names if "proj" in n]
    # print(names)
    # import pdb; pdb.set_trace()

    # for name, module in model.named_modules():
    #     if isinstance(module, bnb.nn.Linear4bit):# or isinstance(module, auto_gptq.nn_modules.qlinear.qlinear_exllamav2.QuantLinear):
    #         names = name.split(".")
    #         lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # if "lm_head" in lora_module_names:  # needed for 16-bit
    #     lora_module_names.remove("lm_head")
    return lora_module_names


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    # import pdb; pdb.set_trace()
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32 for
    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if bf16:
    #             module = module.to(torch.bfloat16)
    #     if "norm" in name:
    #         module = module.to(torch.float32)
    #     if "lm_head" in name or "embed_tokens" in name:
    #         if hasattr(module, "weight"):
    #             if bf16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# create peft config
model = create_peft_model(
   model)#, gradient_checkpointing=False, bf16=False
#)

# Define training args
output_dir = "."
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    #bf16=False,  # Use BF16 if available
    learning_rate=2e-4,
    num_train_epochs=3,
    optim="paged_adamw_8bit", #"adamw_torch" if not mode = 4,8
    gradient_checkpointing=False,
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    remove_unused_columns=False,
)

    # Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the model
trainer.save_model('./pretrained_model')

# trainer.train(resume_from_checkpoint='./checkpoint-500')

# from peft import PeftModel

# # model is the quantized model loaded using load_hf_model in a previous step
# model = PeftModel.from_pretrained(model, './pretrained_model')

import torch
import transformers
from transformers import Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from trl import SFTTrainer
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers import TrainerCallback
from contextlib import nullcontext

from transformers import LlamaForCausalLM, LlamaTokenizer
from preproc_utils import get_preprocessed_samsum, get_preprocessed_debatabase


from summary_metrics import parse_text_to_networkx, node_stance_accuracy, node_stance_f1
from datasets import load_metric

meteor = load_metric('meteor')
rouge = load_metric('rouge')

print("cuda available?", torch.cuda.is_available())

model_id = 'meta-llama/Llama-2-7b-hf'

# hf_auth = <Copy&Paste here your Hugging Face User Access Token>
tokenizer = LlamaTokenizer.from_pretrained(
    model_id,
    # use_auth_token=hf_auth
)
tokenizer.pad_token = "[PAD]"


def compute_node_stance_acc_f1(label_str, pred_str):
    node_accs = []
    node_f1s = []
    for lab, pred in list(zip(label_str, pred_str)):

        gold_graph = parse_text_to_networkx(lab)
        pred_graph = parse_text_to_networkx(pred)
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






# # samsum_dataset = get_preprocessed_samsum(tokenizer, 'validation')
# train_dataset = get_preprocessed_debatabase(tokenizer, "train")
# val_dataset = get_preprocessed_debatabase(tokenizer, "val")
# val_dataset = val_dataset.select(range(2))


# model = LlamaForCausalLM.from_pretrained(
#     model_id,  device_map='auto', torch_dtype=torch.float16,
#     # use_auth_token=hf_auth
# )
# model.resize_token_embeddings(len(tokenizer))

# eval_prompt = """
# Create an Argument Graph from these comments:
# Comment 1: Underground nuclear waste storage means that nuclear waste is stored at least 300m underground. [I1]The harm of a leak 300m underground is significantly limited, if the area has been chosen correctly then there should be no water sources nearby to contaminate. If this is the case, then a leak’s harm would be limited to the layers of sediment nearby which would be unaffected by radiation. By comparison a leak outside might lead to animals nearby suffering from contamination. Further nuclear waste might reach water sources should there be a leak above ground, if it is raining heavily when the leak happens for example. Further, the other options available, such as above ground storage present a potentially greater danger, should something go wrong. This is because it is much easier for nuclear waste to leak radiation into the air. This is problematic because even a hint of radiation may well cause people to panic owing to the damaging and heavily publicised consequences of previous nuclear safety crises. As such, underground storage is safer both directly and indirectly.[1] As well as this, underground storage also prevents nuclear waste or nuclear radiation from reaching other states and as such, results in greater safety across borders.[2] Further, storing all nuclear waste underground means that countries can concentrate their research and training efforts on responding to subterranean containment failures. Focus and specialisation of this type is much more likely to avert a serious release of nuclear material from an underground facility than the broad and general approach that will be fostered by diverse and distinct above-ground storage solutions. [1] “Europe eyes underground nuclear waste repositories.” Infowars Ireland. 20/02/2010 http://info-wars.org/2010/02/20/europe-eyes-underground-nuclear-waste-repositories/[2] “EU Debates Permanent Storage For Nuclear Waste.” 04/11/2010 AboutMyPlanet. http://www.aboutmyplanet.com/environment/eu-debates-permanent-storage-for-nuclear-waste/ [I1]I am not sure how to replace this section. “Leakage” of radioactive material into the air is a minimal danger. The contributor may be referring to the ejection of irradiated dust and other particulates that has occurred when nuclear power stations have suffered explosive containment failures, but this is not comparable to the types of containment failures that might happen in facilities used to store spent nuclear fuel rods and medical waste. One of the more substantial risks presented by underground storage is release of nuclear material into a water source. 

# Comment 2: Even states without nuclear waste programs tend to generate radioactive waste. For example, research and medicine both use nuclear material and nuclear technology. Technologies such as Medical imaging equipment are dependent and the use of radioactive elements. This means that all states produce levels of nuclear waste that need to be dealt with.Moreover, many non-nuclear states are accelerating their programmes of research and investment into nuclear technologies. With the exception of Germany, there is an increasing consensus among developed nations that nuclear power is the only viable method of meeting rising domestic demand for energy in the absence of reliable and efficient renewable forms of power generation. The alternatives to putting nuclear waste in underground storage tend to be based around the reuse of nuclear waste in nuclear power stations. Whilst this is viable in some areas, in countries which lack the technology to be able to do this and in countries which don’t need to rely on nuclear power, this option becomes irrelevant. Further, even this process results in the creation of some nuclear waste, so in countries with the technology to implement such a solution, the disposal of the remaining nuclear waste is still an issue. As such, underground nuclear storage is a necessary method that should be used to dispose of nuclear waste.[1] [1] “The EU’s deep underground storage plan.” 03/11/2010. World Nuclear News. http://www.world-nuclear-news.org/WE_The_EUs_deep_underground_storage_plan_0311101.html 

# Comment 3: Underground nuclear storage is expensive. This is because the deep geological repositories needed to deal with such waste are difficult to construct. This is because said repositories need to be 300m underground and also need failsafe systems so that they can be sealed off should there be a leak. For smaller countries, implementing this idea is almost completely impossible. Further, the maintenance of the facilities also requires a lot of long term investment as the structural integrity of the facilities must consistently be monitored and maintained so that if there is a leak, the relevant authorities can be informed quickly and efficiently. This is seen with the Yucca mountain waste repository site which has cost billions of dollars since the 1990s and was eventually halted due to public fears about nuclear safety.[1] [1] ISN Security Watch. “Europe’s Nuclear Waste Storage Problems.” Oilprice.com 01/06/2010 http://oilprice.com/Alternative-Energy/Nuclear-Power/Europes-Nuclear-Waste-Storage-Problems.html 

# Comment 4: There are new kinds of nuclear reactor such as ‘Integral Fast Reactors’, which can be powered by the waste from normal nuclear reactors (or from uranium the same as any other nuclear reactor). This means that the waste from other reactors or dismantled nuclear weapons could be used to power these new reactors. The Integral Fast Reactor extends the ability to produce energy roughly by a factor of 100. This would therefore be a very long term energy source.[1] The waste at the end of the process is not nearly as much of a problem, as it is from current reactors. Because the IFR recycles the waste hundreds of times there is very much less waste remaining and what there is has a much shorter half-life, only tens of years rather than thousands. This makes storage for the remainder much more feasible, as there would be much less space required.[2] [1] Till, Charles, ‘Nuclear Reaction Why DO Americans Fear Nuclear Power’, PBS, http://www.pbs.org/wgbh/pages/frontline/shows/reaction/interviews/till.html[2] Monbiot, George, ‘We need to talk about Sellafield, and a nuclear solution that ticks all our boxes’, guardian.co.uk, 5 December 2011, http://www.guardian.co.uk/commentisfree/2011/dec/05/sellafield-nuclear-energy-solution 

# Comment 5: France is the largest nuclear energy producer in the world. It generates 80% of its electricity from nuclear power.[1] It is very important to note, therefore, that it does not rely on underground nuclear waste storage. Instead, it relies on above ground, on-site storage. This kind of storage combined with heavy reprocessing and recycling of nuclear waste, makes underground storage unnecessary.[2] As such it seems logical that in most western liberal democracies that are able to reach the same level of technological progress as France, it makes more sense to store nuclear waste above ground. Above ground, checks and balances can be put into place that allow the maintenance of these nuclear storage facilities to be monitored more closely. Furthermore, reprocessing and recycling leads to less wasted Uranium overall. This is important as Uranium, whilst being plentiful in the earth, is often difficult to mine and mill. As such, savings here often significantly benefit things such as the environment and lower the economic cost of the entire operation. [1] BBC News, ‘France nuclear power funding gets 1bn euro boost’, 27 June 2011, http://www.bbc.co.uk/news/world-europe-13924602[2] Palfreman, Jon. “Why the French Like Nuclear Energy.” PBS. http://www.pbs.org/wgbh/pages/frontline/shows/reaction/readings/french.html 
# ---
# Argument Graph:
# """

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=400)[0], skip_special_tokens=True))

# model.train()


# def create_peft_config(model):
#     from peft import (
#         get_peft_model,
#         LoraConfig,
#         TaskType,
#         prepare_model_for_int8_training,
#     )

#     peft_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         inference_mode=False,
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         target_modules=[
#     "q_proj",
#     "up_proj",
#     "o_proj",
#     "k_proj",
#     "down_proj",
#     "gate_proj",
#     "v_proj"
#   ]

#     )

#     # prepare int-8 model for training
#     model = prepare_model_for_int8_training(model)
#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters()
#     return model, peft_config


# # create peft config
# model, lora_config = create_peft_config(model)

# enable_profiler = False
# output_dir = "tmp/llama-output"

# config = {
#     'lora_config': lora_config,
#     'learning_rate': 1e-4,
#     'num_train_epochs': 40,
#     'gradient_accumulation_steps': 2,
#     'per_device_train_batch_size': 1,
#     'per_device_eval_batch_size': 1,
#     'gradient_checkpointing': False,
# }

# # Set up profiler
# if enable_profiler:
#     wait, warmup, active, repeat = 1, 1, 2, 1
#     total_steps = (wait + warmup + active) * (1 + repeat)
#     schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
#     profiler = torch.profiler.profile(
#         schedule=schedule,
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True)


#     class ProfilerCallback(TrainerCallback):
#         def __init__(self, profiler):
#             self.profiler = profiler

#         def on_step_end(self, *args, **kwargs):
#             self.profiler.step()


#     profiler_callback = ProfilerCallback(profiler)
# else:
#     profiler = nullcontext()

# # Define training args
# training_args = Seq2SeqTrainingArguments(
#     output_dir=output_dir,
#     overwrite_output_dir=True,
#     bf16=True,  # Use BF16 if available
#    # evaluation_strategy="epoch",
#     evaluation_strategy="steps",
#     eval_steps=1,
#     predict_with_generate=True,
#     generation_config= transformers.GenerationConfig(max_new_tokens=400),
#     # logging strategies
#     logging_dir=f"{output_dir}/logs",
#     logging_strategy="epoch",
#    # save_strategy="epoch",
#     optim="adamw_torch_fused",
#     max_steps=total_steps if enable_profiler else -1,
#     metric_for_best_model="rouge2_f_measure",
#     greater_is_better=True,
#     load_best_model_at_end=True,
#     **{k: v for k, v in config.items() if k != 'lora_config'}
# )

# # Create Trainer instance
# # data_collator = DataCollatorForTokenClassification(tokenizer)

# trainer = Seq2SeqTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
#     # data_collator=data_collator,
#     callbacks=[profiler_callback] if enable_profiler else [],
# )

# # Start training
# trainer.train(resume_from_checkpoint=False)

# model.save_pretrained(output_dir)

# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=400)[0], skip_special_tokens=True))

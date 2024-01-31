
from datasets import Dataset
from peft import LoraConfig, PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback,  GenerationConfig
from trl import SFTTrainer
import torch
import pandas as pd

from preproc_utils import get_preprocessed_debatabase_sft, get_raw_debatabase
from datasets import load_metric



from evaluate import load

from copy import deepcopy
from summary_metrics import compute_node_stance_acc_f1_ged
import re
import os
import os.path as op

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--lora_dropout", type=float, default=0.5)
parser.add_argument("--learning_rate",type=float,default=1e-4)
parser.add_argument("--weight_decay",type=float, default=0.001)
args = parser.parse_args()

dir_name = "_".join([str(k) + "_" + str(v) for k,v in vars(args).items()])



MULTILEVEL=False

if MULTILEVEL==True:
    scores_dir = op.join("prompting_scores_multilevel", dir_name)
else:
    scores_dir = op.join("prompting_scores", dir_name)

if not op.exists(scores_dir):
    os.mkdir(scores_dir)


meteor = load_metric('meteor')
rouge = load_metric('rouge')


def compute_metrics(predictions, references):

    meteor_output = meteor.compute(predictions=predictions, references=references)
    print("hello")
    print("hello2")
    print(predictions)
    print(references)

    rouge_output = rouge.compute(
         predictions=predictions, references=references, rouge_types=['rouge2'])['rouge2'].mid

    node_acc, node_f1, ged = compute_node_stance_acc_f1_ged(predictions=predictions, references=references)

    return {
        'meteor_score': round(meteor_output['meteor'], 4),
        'rouge2_precision': round(rouge_output.precision, 4),
        'rouge2_recall': round(rouge_output.recall, 4),
        'rouge2_f_measure': round(rouge_output.fmeasure, 4),
        'node stance f1': round(node_f1, 4),
        'node stance acc': round(node_acc, 4),
        "graph edit distance": round(ged, 4)
    }







# eval_callback = EvalCallback()


train_dataset = get_raw_debatabase("train",multilevel=MULTILEVEL)
val_dataset = get_raw_debatabase("val",multilevel=MULTILEVEL)
test_dataset = get_raw_debatabase("test",multilevel=False)
test_dataset_multilevel = get_raw_debatabase("test", multilevel=True)
# # train_dataset = train_dataset.select(range(2))
# val_dataset = val_dataset.select(range(3))



# bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=getattr(torch, 'float16'),
    bnb_4bit_use_double_quant=False,
)

# LoRa
peft_config = LoraConfig(
    # r=8,
    # lora_alpha=32,
    # lora_dropout=0.05,

    lora_alpha=16,
    lora_dropout=args.lora_dropout,
    r=64,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=["q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj"]
)

# Llama 2

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=bnb_config,
)
# model.cuda()
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


prompt_template_long=f"""
[INST]
Main topic: This House believes Europe still needs a constitution

Comment 1: A comprehensive reform of the EU institutional layout is a must given the pressures created by the continuing enlargement process as well as the integration process. The existing EU architecture worked fine for a community of six states, and even for a group of twelve, but it is now desperately out-dated and unsuitable for a Union of 27 or more. For example, the national veto still applies in many areas, meaning one state can block progress even when the other 26 agree. Even when agreement is reached, it is often agonisingly slow and difficult to implement across the whole of the Union, often having to pass through every parliament. As a result EU decision-making has often been criticised as slow, complex and producing too many ‘lowest common denominator’ solutions, therefore Ireland can bring to a halt a vital treaty like Lisbon[1] and the role of the Presidency and ‘foreign minister’ is a compromise that does not result in more unified policy.[2] While still leaving the people feeling distant from the EU’s political processes, undermining legitimacy.[3] A Constitutional Treaty is the only comprehensive tool that exists right now in order to allow for this necessary overall reform.  [1] BBC News, ‘Ireland rejects EU reform treaty’, 13 June 2008, http://news.bbc.co.uk/1/hi/7453560.stm[2] Bellotti, Sarah M., and Dale, Reginald, ‘U.S. Media Snubs New EU Leaders’, Center for Strategic & International Studies, http://csis.org/blog/us-media-snubs-new-eu-leaders[3] Renda, Andrea, ‘Policy-Making in the EU; Achievements, Challenges and Proposals for Reform’, Centre for European Policy Studies, 2009, www.ceps.eu/files/book/1854.pdf

Comment 2: The European Union should be wary of adopting a European Constitution as many states may not be able to abide by its terms. The reason why Greece is in so much financial trouble is its unwillingness to abide by the European Growth and Stability Pact, however others, Germany and France had already broken the pact.[1] Such a failure to abide by the rules with a constitution, something which is meant to be at the heart of the state, would greatly damage European credibility and would practically rule out the possibility of more comprehensive change in the future.Accession countries have shown little interest in the Constitutional Treaty overall, given a series of other more immediate concerns. Therefore a constitution is unneeded in order for the EU to develop, enlarge or prosper. It can only lose if it created a constitution which turned out a disaster.[1] Aznar, José María, ‘Europe must reset the clock on stability and growth’, FT.com, 16 May 2010, http://www.ft.com/cms/s/0/6e07c4f0-6115-11df-9bf0-00144feab49a.html#axzz1a1I5v8kb

Comment 3: Since the Maastricht Treaty, the citizens of EU member states have possessed parallel citizenship of the EU. However, European citizens do not identify themselves with the EU in the way that citizens of the USA self-identify as American. An important part of the patriotism of Americans is ‘constitutional patriotism;’ pride in their constitution and civic institutions. The European Union aims to bring about ever closer union between the peoples of Europe. It should foster a shared sense of ‘European identity’ by adopting a constitution, in which every citizen of the EU can take pride.

Comment 4: The current treaty-basis for the European Union is enormous, ambiguous and extremely complicated. The existing treaties regulate multiple levels from the constitutional to detailed market regulations. As a result of this individuals cannot easily read and understand the treaties as a US citizen for example.[1] It is difficult to keep track of each new Treaty that amends the pre-existing treaties. The adoption of a shorter, clearer document will make the EU much more ‘user friendly.’ The EU currently suffers from the fact that many of its citizens do not know what it is or what it does; EU citizens either do not know where to look for this information or are deterred and intimidated by the size of the Treaty of Rome and the Maastricht Treaty. Having an easily digestible constitution will mean that the EU’s citizens can easily find out what the EU is and what it does.[1] Gjørtler, Peter, ‘ Lisbon Treaty - the Reform Treaty of the European Union’, grayston & company, November 2009 http://www.graystoncompany.com/gco/articles/G&C_article_Lisbon_Treaty_PG.pdf

Comment 5: We already have such constitutional documents – the Treaty of Rome, the Maastricht Treaty and most importantly the Lisbon treaty from very recently (2009). The powers of, and relationships between the different institutional actors are clearly defined in the existing treaties. Just because the EU has expanded to incorporate new member states does not mean it needs a constitution. The Treaty of Nice was meant to have made the necessary amendments to facilitate enlargement. If it has failed, then we can simply amend the existing treaties again.

Comment 6: The European Court of Justice (ECJ) has long treated the founding treaties as the constitutional documents of the European Union. Many commentators have noted the efforts of the ECJ to “constitutionalise” many principles – such as the direct effect and supremacy of Community law over the domestic laws of member states and the increasing protection of human rights – The ECJ is often overstepping its bounds when it comes to applying and interpreting the treaties.[1] The ECJ has often been accused of “judicial activism” in over-stepping the legitimate boundaries of courts in a democracy. By enshrining much of this creative jurisprudence in a democratically ratified constitution, the EU can assert and emphasise its status as a democratic entity, rather than an elite-driven process separate from the citizens of Europe.[1] Roberts, Linda, ‘The CARICOM Single Market and Economy and the Caribbean Court of Justice’, Southampton Working Papers in European Law, 2007/1, http://www.eulaw.soton.ac.uk/elpub/SWPEL3_Roberts.pdf

Comment 7: A European constitution is a first step on a slippery slope towards a United States of Europe. Such a European superstate is widely opposed by citizens of all EU members, not least because it would be undemocratic, unaccountable and remote. Many EU citizens already believe this is the case. In Britain polls regularly show that far from wanting deeper integration the country is in favour of leaving the EU.[1]  As has already been shown members do not consider themselves ‘European’ nearly as much as they do their own national identity. [2][1] The Democracy Movement Surrey, ‘The EU - Superstate or Free Trade Partner? We Can Leave.’ 2007 http://www.democracymovementsurrey.co.uk/canweleave.html[2] Turmo, Ivan and Bradley, Simon, ‘Poll reveals European mindset among Swiss’, swissinfo.ch, 11 August 2010, http://www.swissinfo.ch/eng/politics/foreign_affairs/Poll_reveals_European_mindset_among_Swiss.html?cid=22188596 

[/INST]
[SOG]
Comment 1 (supports main topic): A comprehensive reform of the EU institutional layout is a must

Comment 2 (attacks main topic): Adopting a European Constitution and failing to abide by it would be a big and challenging failure

Comment 3 (supports main topic): A EU constitution will foster a “European identity”

Comment 4 (supports main topic): The current treaty-basis for the European Union is enormous, ambiguous and extremely complicated

Comment 5 (attacks main topic): There already are constitutional documents

Comment 6 (supports main topic): The ECJ has often been accused of over-stepping the legitimate boundaries of 

Comment 7 (attacks main topic): A EU Constitution will lead to a superstate, which is undesirable at the moment
[EOG]
[INST]
{{comments}}
[/INST]
"""


prompt_template=f"""
[INST]
Main topic: This House believes Europe still needs a constitution

Comment 1: A comprehensive reform of the EU institutional layout is a must given the pressures created by the continuing enlargement process as well as the integration process. The existing EU architecture worked fine for a community of six states, and even for a group of twelve, but it is now desperately out-dated and unsuitable for a Union of 27 or more. For example, the national veto still applies in many areas, meaning one state can block progress even when the other 26 agree. Even when agreement is reached, it is often agonisingly slow and difficult to implement across the whole of the Union, often having to pass through every parliament. 

Comment 2: The European Union should be wary of adopting a European Constitution as many states may not be able to abide by its terms. The reason why Greece is in so much financial trouble is its unwillingness to abide by the European Growth and Stability Pact, however others, Germany and France had already broken the pact.[1] Such a failure to abide by the rules with a constitution, something which is meant to be at the heart of the state, would greatly damage European credibility and would practically rule out the possibility of more comprehensive change in the future.

Comment 3: Since the Maastricht Treaty, the citizens of EU member states have possessed parallel citizenship of the EU. However, European citizens do not identify themselves with the EU in the way that citizens of the USA self-identify as American. An important part of the patriotism of Americans is ‘constitutional patriotism;’ pride in their constitution and civic institutions. The European Union aims to bring about ever closer union between the peoples of Europe. It should foster a shared sense of ‘European identity’ by adopting a constitution, in which every citizen of the EU can take pride.
[/INST]
[SOG]
Comment 1 (supports main topic): A comprehensive reform of the EU institutional layout is a must

Comment 2 (attacks main topic): Adopting a European Constitution and failing to abide by it would be a big and challenging failure

Comment 3 (supports main topic): A EU constitution will foster a “European identity”
[EOG]
[INST]
{{comments}}
[/INST]
"""


prompt_template_multilevel=f"""
[INST]
Main topic: This House would lease Crimea to Russia

Comment 1: There is a lot more at stake than just the Crimean peninsula. While suggestions that it may destroy the whole international system are hyperbole the territory becoming part of Russia would be the most major territorial change in Europe since the unification of Germany and breakup of the USSR both of which were peaceful and mutually agreed events.  

Comment 2: The big advantage of a lease is that it maintains the territorial status quo while giving Russia what it wants. If the concern is about the legal order and sovereignty of states then a lease provides the answer because the actual sovereignty over the territory is not handed over, merely the control over the territory and functions of that territory are. 

Comment 3: It is hard to see why Ukraine would be willing to sign a lease with Russia when Russia has already proven it will not stick to the terms of its lease. Russia signed agreements in 1997 that recognised Crimea as a part of Ukraine in return for a lease on the base of the Russian Black Sea Fleet.

Comment 4: While of the core points of sovereignty is that is indivisible this has not stopped the existence of other similar deals happening in the past.
[/INST]
[SOG]
Comment 1 (attacks main topic): The crisis affects more than just Crimea

Comment 2 (attacks Comment 1): The big advantage of a lease is that it maintains the territorial status quo while giving Russia what it wants.

Comment 3 (attacks main topic): Why would Ukraine trust a lease when the previous one was violated?

Comment 4 (supports main topic): There are precedents
[EOG]
[INST]
{{comments}}
[/INST]
"""




def do_prompting():


    model.eval()

    gold_texts = []
    generated_texts = []

    generation_config=GenerationConfig(
        do_sample=False,
        max_new_tokens=512,        
    )

    input_token_lengths = []
    output_token_lengths = []

    failed_count = 0
    succeeded_count = 0

    for sample in test_dataset:

        gold_texts.append(sample["summaries"])
        input_text = prompt_template.format(comments=sample["comments"])

        input_tok = tokenizer.encode(input_text, return_tensors="pt").cuda()
        output_tok = model.generate(input_ids=input_tok, generation_config=generation_config)
  

        generated_text = tokenizer.decode(output_tok[0])

        generated_text = generated_text.split("[EOG]",1)[1]

        all_outputs = re.findall(r"\[SOG\](.*?)\[EOG\]", generated_text, flags=re.DOTALL)

        if len(all_outputs) < 1:

            if "[SOG]" in generated_text:
                output_text = re.split(r"\[SOG\]",generated_text)[1]
                output_text = re.split(r"[\[\]]", output_text)[0]

                succeeded_count+=1
                print("**********NO EOG*************")
                print(output_text)
                
            else:
                output_text = "fail"

                failed_count +=1 
                print("*********FAILED********")
                print(generated_text)
    
        else:
            output_text = all_outputs[0]
            succeeded_count +=1 
            print("***********YES EOG************")
            print(output_text)
            # print(output_text)
        
        generated_texts.append(output_text)

        input_token_lengths.append(len(input_tok[0]))
        output_token_lengths.append(len(output_tok[0]))

    # del(model2)

    print(f"max input token length: {max(input_token_lengths)}")
    print(f"max output token length: {max(output_token_lengths)}")

    print(f"failed count: {failed_count}")
    print(f"succeeded count: {succeeded_count}")




    metrics = compute_metrics(predictions=generated_texts, references=gold_texts)


    sample_output = generated_texts[-1]

    with open(f"scores_single_prompting/sample_output.txt","w") as sample_file:
        sample_file.write(sample_output + "\n\n")
    scores_df = pd.DataFrame([metrics])
    scores_df.to_csv(f"scores_single_prompting/single_level_results.csv")


def do_prompting_multilevel():


    model.eval()

    gold_texts = []
    generated_texts = []

    generation_config=GenerationConfig(
        do_sample=False,
        max_new_tokens=512,        
    )

    input_token_lengths = []
    output_token_lengths = []

    failed_count = 0
    succeeded_count = 0

    for sample in test_dataset_multilevel:

        gold_texts.append(sample["summaries"])
        input_text = prompt_template_multilevel.format(comments=sample["comments"])

        input_tok = tokenizer.encode(input_text, return_tensors="pt").cuda()
        output_tok = model.generate(input_ids=input_tok, generation_config=generation_config)
  

        generated_text = tokenizer.decode(output_tok[0])

        generated_text = generated_text.split("[EOG]",1)[1]

        all_outputs = re.findall(r"\[SOG\](.*?)\[EOG\]", generated_text, flags=re.DOTALL)

        if len(all_outputs) < 1:

            if "[SOG]" in generated_text:
                output_text = re.split(r"\[SOG\]",generated_text)[1]
                output_text = re.split(r"[\[\]]", output_text)[0]

                succeeded_count+=1
                print("**********NO EOG*************")
                print(output_text)
                
            else:
                output_text = "fail"

                failed_count +=1 
                print("*********FAILED********")
                print(generated_text)
    
        else:
            output_text = all_outputs[0]
            succeeded_count +=1 
            print("***********YES EOG************")
            print(output_text)
            # print(output_text)
        
        generated_texts.append(output_text)

        input_token_lengths.append(len(input_tok[0]))
        output_token_lengths.append(len(output_tok[0]))

    # del(model2)

    print(f"max input token length: {max(input_token_lengths)}")
    print(f"max output token length: {max(output_token_lengths)}")

    print(f"failed count: {failed_count}")
    print(f"succeeded count: {succeeded_count}")




    metrics = compute_metrics(predictions=generated_texts, references=gold_texts)


    sample_output = generated_texts[-1]

    with open(f"scores_single_prompting/sample_output_multilevel.txt","w") as sample_file:
        sample_file.write(sample_output + "\n\n")
    scores_df = pd.DataFrame([metrics])
    scores_df.to_csv(f"scores_single_prompting/multi_level_results.csv")




do_prompting()
do_prompting_multilevel()
#trainer.model.save_pretrained('llama-2-7b-nmt')


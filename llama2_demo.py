from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import pandas as pd

e2e_df = pd.read_csv("debatabase_data/end_to_end.csv")


model_name = "meta-llama/Llama-2-70b-hf"

model = AutoModelForCausalLM.from_pretrained(model_name)
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)


def mod_gen(input_ids, max_new_tokens=200):
    generation_output = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(temperature=1.0, top_p=1.0, top_k=50, num_beams=1),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens
    )
    for seq in generation_output.sequences:
        output = tokenizer.decode(seq)
        print(output)

def generate(instruction):
    # prompt = "### Human: "+instruction+"### Assistant: "
    inputs = tokenizer(instruction, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    mod_gen(input_ids, 50)
    import pdb; pdb.set_trace()


# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )


prompt = "Instructions: For each comment below, write a summary and state whether it agrees or disagrees with the main topic. \n\n"

model_input = prompt + e2e_df.iloc[0]["comments"]

generate(model_input)
import pdb; pdb.set_trace()

# sequences = pipeline(
#     model_input,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=2000,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")


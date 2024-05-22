import transformers
import torch
import csv
import pdb
import json
from tqdm import tqdm

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        )

csv_data = csv.reader(open('/home/darshshah/synthetic_data/'
    'FinanceBench_Apr24/financebench_sample_150.csv','r'))

synthetic_data = {}

all_lines = []

output_writer = csv.writer(open('/home/darshshah/synthetic_data/FinanceBench_Apr24/'
    'syntactic_augmented_financebench_sample_150.csv','w'))

json_data = []

for ind,line in tqdm(enumerate(csv_data)):
    output_writer.writerow(line)
    if ind == 0:
        continue

    messages = [{"role": "system", "content": "You are a synthetic data generator. For the input question. Please re-write it in a different way. Please note the exact meaning of the generated question should be exactly the same."},
                {"role": "user", "content": line[5]},]
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a synthetic data generator. For the input question, please re-write it in a different way. Please note the exact meaning of the generated question should be exactly the same. <|eot_id|><start_header_id|>user<|end_header_id|>'%s'."%line[5]
    prompt+= '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nRewritten question is: '
    json_data.append({'prompt':prompt,'question':line[5]})
    continue

    if line[5] not in synthetic_data:
        synthetic_data[line[5]] = []

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )

    terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]


    for t in [0.3, 0.6, 0.9]:
        outputs = pipeline(
                prompt,
                max_new_tokens=25600,
                eos_token_id=terminators,
                do_sample=True,
                temperature=t,
                top_p=0.9,
        )
        candidate_output = outputs[0]["generated_text"][len(prompt):]
        if "\n\n" in candidate_output:
            candidate_output = candidate_output.split("\n\n")[1]
        if candidate_output not in synthetic_data[line[5]] and candidate_output != line[5]:
            synthetic_data[line[5]].append(candidate_output)

    for candidate in synthetic_data[line[5]]:
        output_writer.writerow(line[:5] + [candidate] + line[6:])

json.dump(json_data,open("/home/darshshah/synthetic_data/data/type1.json","w"))

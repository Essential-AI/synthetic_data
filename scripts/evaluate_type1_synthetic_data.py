import transformers
import torch
import csv
import pdb
import json
from tqdm import tqdm

input_data  = json.load(open('../data/type1.json','r'))
output_data = json.load(open('/home/darshshah/synthetic_data/scripts/lm_eval_tasks/type1/outputs/pretrained=gpt4-turbo_type1.jsonl','r'))

eval_data = []

for input_d,output_d in zip(input_data, output_data):

    human_written_prompt = output_d['doc']['prompt'].replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>","").replace("<|eot_id|><start_header_id|>user<|end_header_id|>","").replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>","")
    machine_output = output_d['resps'][0][0].strip()

    grammar_prompt  = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "You are a data quality evaluator. For the candidate output, generate a YES or NO on whether it is grammatically correct.<|eot_id|><|start_header_id|>user<|end_header_id|>Candidate Output:'%s'\n\n"%(machine_output)+\
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerated Response:"
    fidelity_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "You are a data quality evaluator. For the requested prompt, generate a YES or NO on whether the candidate output follows the instruction entirely."\
            + "<|eot_id|><|start_header_id|>user<|end_header_id|>" + "Requested Prompt:'%s'\n\nCandidate Output:'%s'\n\n" %(human_written_prompt,machine_output) + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerated Response:"
    eval_data.append({'grammar_prompt':grammar_prompt, 'fidelity_prompt':fidelity_prompt})

json.dump(eval_data, open("../data/gpt4_eval_type1.json","w"), indent=2)

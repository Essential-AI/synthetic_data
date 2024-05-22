import transformers
import torch
import csv
import pdb
import json
from tqdm import tqdm

input_data  = json.load(open('../data/type3.json','r'))
output_data = json.load(open('/home/darshshah/synthetic_data/scripts/lm_eval_tasks/type3/outputs/pretrained=gpt4-turbo_type3.jsonl','r'))

eval_data = []

for input_d,output_d in zip(input_data, output_data):

    human_written_prompt = output_d['doc']['prompt'].replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>","").replace("<|eot_id|><start_header_id|>user<|end_header_id|>","").replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>","")
    machine_output = output_d['resps'][0][0].strip()

    grammar_prompt  = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "You are a data quality evaluator. For the candidate output, generate a YES or NO on whether it is grammatically correct.<|eot_id|><|start_header_id|>user<|end_header_id|>Candidate Output:'%s'\n\n"%(machine_output)+\
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerated Response:"
    fidelity_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "You are a data quality evaluator. For the requested prompt, generate a YES or NO on whether the candidate output follows the instruction entirely."\
            + "<|eot_id|><|start_header_id|>user<|end_header_id|>" + "Requested Prompt:'%s'\n\nCandidate Output:'%s'\n\n" %(human_written_prompt,machine_output) + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerated Response:"

    eval_dict = {'grammar_prompt':grammar_prompt, 'fidelity_prompt':fidelity_prompt}
    for sampled_key in ['sampled_year','sampled_company']:
        eval_dict[sampled_key + '_prompt'] = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a data quality evaluator. For the candidate output, generate a YES or NO on whether it refers to a specific entity.<|eot_id|><|start_header_id|>user<|end_header_id|>Requested '%s' is: '%s'\n\nCandidate Output:'%s'\n\n"%(sampled_key,input_d[sampled_key],machine_output)
    eval_dict['section_prompt'] = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a data quality evaluator. For the candidate output, generate a YES or NO on whether it can be correctly answered from the text section provided.<|eot_id|><|start_header_id|>user<|end_header_id|>Text Section is: '%s'\n\nCandidate Output: '\%s'\n\n"%(input_d['text_section'], machine_output)
    eval_data.append(eval_dict)

json.dump(eval_data, open("../data/gpt4_eval_type3.json","w"), indent=2)

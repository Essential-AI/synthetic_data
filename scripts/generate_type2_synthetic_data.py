import transformers
import torch
import csv
import pdb
import pandas as pd
import yfinance as yf
import random
import pickle
import os
import json
from yahoo_fin import stock_info as si
from tqdm import tqdm

random.seed(42)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        )

csv_data = csv.reader(open('/home/darshshah/synthetic_data/'
    'FinanceBench_Apr24/financebench_sample_150.csv','r'))

num_companies = 10
num_years     = 10

synthetic_data = {}

all_lines = []

json_data = []

nasdaq_tickers = pd.DataFrame( si.tickers_nasdaq() )[0].values.tolist()
if os.path.exists("company_names.p"):
    company_names = pickle.load(open("company_names.p","rb"))
else:
    company_names  = {}
    for nasdaq_ticker in tqdm(nasdaq_tickers):
        company = yf.Ticker(nasdaq_ticker)
        try:
            company_name = company.info['longName']
            if 'Inc' in company_name:
                company_name = company_name[:company_name.rfind("Inc")].strip()
            if company_name.endswith(","):
                company_name = company_name[:company_name.rfind(",")]
            company_names[company_name.strip()] = nasdaq_ticker
        except:
            continue
    pickle.dump(company_names, open("company_names.p","wb"))

company_names = list(company_names.keys())

output_writer = csv.writer(open('/home/darshshah/synthetic_data/FinanceBench_Apr24/'
    'fields_augmented_financebench_sample_150.csv','w'))

for ind,line in tqdm(enumerate(csv_data)):
    if ind == 0:
        continue

    company, year = line[1].split("_")[:2]
    year          = int(year[:4])

    sampled_companies = random.choices(company_names, k=num_companies)
    for sampled_company in tqdm(sampled_companies):
        candidate_years = [y for y in range(2010,2025) if y!=year]
        sampled_years   = random.sample(candidate_years, k=num_years)
        if ":" in line[5]:
            actual_question = line[5][line[5].rfind(":")+1:].strip()
        else:
            actual_question = line[5]
        for sampled_year in sampled_years:
            messages = [{"role": "system", "content": "You are a synthetic data generator. Follow instructions to generate data similar to the inputs."},
                    {"role": "user", "content": "The \"QUESTION\" %s is on company %s in year %d. Rewrite this question for company %s in year %d." 
                        %(actual_question,company,year,sampled_company,sampled_year)},]

            prompt_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a synthetic data generator. Follow instructions to generate data similar to the inputs.<|eot_id|><|start_header_id|>user<|end_header_id|> The QUESTION '%s' is on company %s in year %d. Rewrite this question for company %s in year %d.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerated question is :"%(actual_question,company,year,sampled_company,sampled_year)
            json_data.append({'prompt':prompt_text,'question':actual_question,'sampled_company':sampled_company,'sampled_year':sampled_year})
            continue
            prompt = pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                    )

            terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]


            for t in [0.9]:
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
                #print(line[5]+"\n", candidate_output, sampled_company, sampled_year)
                output_writer.writerow(line[:1] + [sampled_company + "_" + str(sampled_year)] + line[2:3] \
                        + [sampled_year] + line[4:5] + [candidate_output] + line[6:])

json.dump(json_data,open("/home/darshshah/synthetic_data/data/type2.json","w"))

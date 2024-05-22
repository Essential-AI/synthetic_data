import transformers
import torch
import csv
import pdb
import pandas as pd
import yfinance as yf
import random
import pickle
import os
import urllib
import json

from sec_api import QueryApi
from yahoo_fin import stock_info as si
from tqdm import tqdm
from bs4 import BeautifulSoup


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

num_companies = 2
num_years     = 3
SEC_PATH      = "/home/darshshah/synthetic_data/SECEdgar/sec-edgar-filings/"


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


company_names = os.listdir(SEC_PATH)

output_writer = csv.writer(open('/home/darshshah/synthetic_data/FinanceBench_Apr24/'
    'text_augmented_financebench_sample_150.csv','w'))

doc_queries = ['10K','10Q']

for ind,line in tqdm(enumerate(csv_data)):
    if ind == 0:
        continue

    company, year = line[1].split("_")[:2]
    year          = int(year[:4])
    if ":" in line[5]:
        actual_question = line[5][line[5].rfind(":")+1:]
    else:
        actual_question = line[5]

    sampled_companies = random.choices(company_names, k=num_companies)
    for sampled_company in tqdm(sampled_companies):
        sec_files = [os.path.join(root, file) \
                for root, dirs, files in \
                os.walk(os.path.join(SEC_PATH,sampled_company)) for file in files]
        if len(sec_files) <= num_years:
            continue
        sampled_sec_files = random.sample(sec_files, k=num_years)
        for sampled_sec_file in sampled_sec_files:
            try:
                soup = BeautifulSoup(open(sampled_sec_file, "r"), 'html.parser')
            except:
                continue
            try:
                sections = soup.find('body').text.split('\n\n\n\n\n')
            except:
                continue
            all_sections = []
            for section in sections:
                if len(section)<500:
                    continue
                if len(section)>5000:
                    continue
                all_sections.append(section)
            if len(all_sections) == 0:
                continue
            sections = all_sections
            text_section = random.choice(sections)
            sampled_year = soup.text[soup.text.find('FILED AS OF DATE:'):].split("\t\t")[1].split('\n')[0][:4]
            if not soup.title:
                continue
            sampled_type = soup.title.text.replace('-','')

            messages = [{"role": "system", "content": "You are a synthetic data generator. Follow instructions to generate data similar to the inputs."},
                    {"role": "user", "content": "The question \"%s\" is on company %s in year %d using extracted span \"%s\". "
                        "Generate a new question for company %s in year %s which can be answered by the extracted span \"%s\"" 
                        %(actual_question,company,year,line[6],sampled_company,sampled_year,text_section)},]

            prompt_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a synthetic data generator. Follow instructions to generate data similar to the inputs.<|eot_id|><|start_header_id|>user<|end_header_id|> The question '%s' is on company '%s' in year '%d' using extrcted span '%s'. Generate a new question for company '%s' in year '%s' which can be answered by the extracted span '%s'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerated questions is :"%(actual_question,company,year,line[6],sampled_company,sampled_year,text_section)
            json_data.append({'prompt':prompt_text,'question':actual_question, 'text_section':text_section, 'sampled_company':sampled_company, 'sampled_year':sampled_year})
            print(len(json_data))
            continue
            1/0

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
                new_line = list(line)
                new_line[1] = sampled_company + "_" + sampled_year + "_" + sampled_type
                new_line[2] = sampled_sec_file
                new_line[3] = candidate_output
                new_line[5] = ""
                new_line[6] = text_section
                new_line[7] = ""
                output_writer.writerow(new_line)

json.dump(json_data, open("/home/darshshah/synthetic_data/data/type3.json","w"), indent=2)

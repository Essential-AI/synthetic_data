import pickle

from tqdm import tqdm
from sec_edgar_downloader import Downloader

dl = Downloader("Essential.ai", "darsh@essential.ai", "/home/darshshah/synthetic_data/SECEdgar")
company_names = pickle.load(open("company_names.p","rb"))
equity_ids    = list(company_names.values())
for equity_id in tqdm(equity_ids):
    for filing_type in tqdm(['10-K','10-Q']):
        try:
            dl.get(filing_type, equity_id)
        except:
            continue

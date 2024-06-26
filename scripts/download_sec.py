import pickle

from tqdm import tqdm
from sec_edgar_downloader import Downloader

dl = Downloader("Essential.ai", "darsh@essential.ai", "/dev/shm/SECEdgar")
company_names = pickle.load(open("company_names.p","rb"))
equity_ids    = list(company_names.values())
for equity_id in tqdm(equity_ids):
    for filing_type in tqdm(['10-K','10-Q']):
        try:
            dl.get(filing_type, equity_id, after="2023-01-01", before="2023-12-31")
        except:
            continue

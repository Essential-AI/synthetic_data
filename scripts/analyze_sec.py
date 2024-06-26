from bs4 import BeautifulSoup
from tqdm import tqdm
import sys
import pdb


def extract_meaningful_chunks(soup):

    chunks = []
    for tag in tqdm(soup.find_all(['h1','h2','h3','h4','h5','h6','p','TEXT'])):
        candidate_text = tag.get_text(strip=True)
        if candidate_text == '':
            continue
        if candidate_text.count(' ')/len(candidate_text) < 0.01:
            continue
        if candidate_text.count('@')/len(candidate_text) > 0.01:
            continue
        chunks.append(candidate_text)

    for table in tqdm(soup.find_all('table')):
        table_data = []
        for row in table.find_all('tr'):
            row_data = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
            if ''.join(row_data) == '':
                continue
            table_data.append(row_data)
        if len(table_data) == 0:
            continue
        chunks.append(table_data)

    return chunks

file_name = sys.argv[1] if len(sys.argv) >= 2 else '/dev/shm/SECEdgar/sec-edgar-filings/AAL/10-K/0000006201-23-000018/full-submission.txt'
soup = BeautifulSoup(open(file_name).read(), 'lxml')
chunks = extract_meaningful_chunks(soup)

pdb.set_trace()

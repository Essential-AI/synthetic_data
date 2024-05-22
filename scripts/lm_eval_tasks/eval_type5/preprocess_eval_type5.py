import re

def doc_to_text(doc):

    return doc['sampled_company_prompt']

def doc_to_target(doc):
    
    return 0

def doc_to_choice(doc):

    return ['YES', 'NO']

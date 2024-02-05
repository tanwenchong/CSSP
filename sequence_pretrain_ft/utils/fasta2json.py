import os
import json
import pandas as pd

from tqdm import tqdm
from Bio import SeqIO
import random

path='protein_data/sequence/uniprot_sprot.fasta'

parser=SeqIO.parse(path,'fasta')

data_all=[]

maxn=10000

for item in parser:
    seq_=str(item.seq)
    i=random.randint(0,100)
    tp=dict()
    tp['sequence']=seq_[i:i+200]
    if len(tp['sequence'])>100:
        data_all.append(tp)
        if len(data_all)>maxn:
            break


json_data=json.dumps(data_all)

with open('protein_data/sequence/debug.json', 'w') as f:
    f.write(json_data)



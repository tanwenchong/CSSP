import os
import json
import pandas as pd

from tqdm import tqdm

def process(data):
    if pd.isna(data):
        return ''
    else:
        return data


def csv2json(path, chain):
    df=pd.read_csv(path)
    seqs=df['sequence'].values
    pdbs=df['pdb'].values

    data_all=[]

    for i in tqdm(range(len(df))):
        tp_=dict()
        tp_['sequence']=process(seqs[i])
        tp_['pdb_file']=process(pdbs[i])
        data_all.append(tp_)

    json_data=json.dumps(data_all)

    with open('../data/infer/bcell/{}.json'.format(chain), 'w') as f:
        f.write(json_data)

chain='heavy'
path='/code/test/struct_downstream/ATUE/data/processed/bcell_debug.csv'
csv2json(path, chain)
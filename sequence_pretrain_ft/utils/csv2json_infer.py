import os
import json
import pandas as pd

from tqdm import tqdm

if 0:
    # H_P_T
    def process(data):
        if pd.isna(data):
            return ''
        else:
            return data

    def csv2json(path, chain):
        df=pd.read_csv(path)
        seqs=df['sequence_{}'.format(chain)].values

        data_all=[]

        for i in tqdm(range(len(df))):
            tp_=dict()
            tp_['sequence']=process(seqs[i])
            tp_['pdb_file']='{}_{}.pdb'.format(i,chain)
            data_all.append(tp_)

        json_data=json.dumps(data_all)

        with open('../data/infer/AbPROP/TMID/{}.json'.format(chain), 'w') as f:
            f.write(json_data)

    def csv2json_paired(path, chain='paired'):
        df=pd.read_csv(path)
        seqsh=df['sequence_heavy'].values
        seqsl=df['sequence_light'].values

        data_all=[]

        for i in tqdm(range(len(df))):
            tp_=dict()
            tp_['sequence_heavy']=process(seqsh[i])
            tp_['sequence_light']=process(seqsl[i])
            tp_['pdb_file']='{}_paired.pdb'.format(i)
            data_all.append(tp_)

        json_data=json.dumps(data_all)

        with open('../data/infer/AbPROP/TMID/{}.json'.format(chain), 'w') as f:
            f.write(json_data)

    path='/code/test/downstream_struct/AbPROP/data/processed/TMID/merged_data.csv'

    chain='paired'
    csv2json_paired(path, chain)

    chain='heavy'
    csv2json(path, chain)

    chain='light'
    csv2json(path, chain)
    
if 1:
    # ALBASE
    def process(data):
        if pd.isna(data):
            return ''
        else:
            return data

    def csv2json(path, chain):
        df=pd.read_csv(path)
        seqs=df['sequence'].values

        data_all=[]

        for i in tqdm(range(len(df))):
            tp_=dict()
            tp_['sequence']=process(seqs[i])
            tp_['pdb_file']='{}_{}.pdb'.format(i,chain)
            data_all.append(tp_)

        json_data=json.dumps(data_all)

        with open('../data/infer/AbPROP/ALBASE/{}.json'.format(chain), 'w') as f:
            f.write(json_data)

    path='/code/test/downstream_struct/AbPROP/data/processed/ALBASE/merged_data.csv'

    chain='light'
    csv2json(path, chain)
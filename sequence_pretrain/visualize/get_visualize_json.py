import os
import json
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

def process(data):
    if pd.isna(data):
        return ''
    else:
        return data

if 0:
    chain='heavy'
    test_files=np.load('../data/{}_test.npy'.format(chain),allow_pickle=True)
    
    path='antibody_data/processed_dataset/sequence/unpaired_heavy_rep_sequences_all_0.6_dedup.csv'

    df=pd.read_csv(path) #['sequence', 'germline', 'cdr1', 'cdr2', 'cdr3', 'species', 'file_name']
    file_names=df['file_name'].values
    
    mask=np.zeros(len(df)).astype(bool)
    for file in tqdm(test_files):
        mask |= (df['file_name']==file).values
        print(mask.sum())
    print(mask.sum())
    df=df[mask]
    
    df.index=np.arange(len(df))
    
    data_test=[]
    species=df['species'].unique()
    for spe in species:
        seqs=df[df['species']==spe]['sequence'].values
        if len(seqs)>10000:
            seqs=random.sample(seqs.tolist(), 10000)
        for i in tqdm(range(len(seqs))):
            tp_=dict()
            tp_['sequence']=seqs[i]
            tp_['species']=spe
            data_test.append(tp_)
    
    print(len(data_test))
    json_data=json.dumps(data_test)

    with open('antibody_pretrain/sequence_pretrain/dedup/{}.json'.format(chain), 'w') as f:
        f.write(json_data)


if 1:
    chain='paired'
    test_files=np.load('../data/{}_test.npy'.format(chain),allow_pickle=True)
    
    path='antibody_data/processed_dataset/sequence/paired_rep_sequences_all_0.7.csv'

    df=pd.read_csv(path) #['sequence', 'germline', 'cdr1', 'cdr2', 'cdr3', 'species', 'file_name']
    
    mask=np.zeros(len(df)).astype(bool)
    for file in tqdm(test_files):
        mask |= (df['file_name']==file).values
    print(mask.sum())
    df=df[mask]
    
    df.index=np.arange(len(df))
    
    data_test=[]
    species=df['species'].unique()
    for spe in species:
        seqsh=df[df['species']==spe]['sequence_heavy'].values
        seqsl=df[df['species']==spe]['sequence_light'].values
        seqs=[]
        for i in range(len(seqsh)):
            seqs.append((seqsh[i],seqsl[i]))
        if len(seqs)>10000:
            seqs=random.sample(seqs, 10000)
        for i in tqdm(range(len(seqs))):
            tp_=dict()
            tp_['sequence_heavy']=seqs[i][0]
            tp_['sequence_light']=seqs[i][1]
            tp_['species']=spe
            data_test.append(tp_)
    
    print(len(data_test))
    json_data=json.dumps(data_test)

    with open('antibody_pretrain/sequence_pretrain/dedup/{}.json'.format(chain), 'w') as f:
        f.write(json_data)

if 0:
    chain='paired'
    train_files=np.load('data/{}_train.npy'.format(chain),allow_pickle=True)
    test_files=np.load('data/{}_test.npy'.format(chain),allow_pickle=True)
    
    path='antibody_data/processed_dataset/sequence/paired_rep_sequences_all_0.7.csv'

    df=pd.read_csv(path) #['sequence', 'germline', 'cdr1', 'cdr2', 'cdr3', 'species', 'file_name']
    
    data_train=[]
    for file in tqdm(train_files):
        mask_=(df['file_name']==file)
        
        seqhs=df['sequence_heavy'][mask_].values
        cdr1hs=df['cdr1_heavy'][mask_].values
        cdr2hs=df['cdr2_heavy'][mask_].values
        cdr3hs=df['cdr3_heavy'][mask_].values
        seqls=df['sequence_light'][mask_].values
        cdr1ls=df['cdr1_light'][mask_].values
        cdr2ls=df['cdr2_light'][mask_][mask_].values
        cdr3ls=df['cdr3_light'][mask_].values

        for i in range(len(seqhs)):
            tp_=dict()
            tp_['sequence_heavy']=process(seqhs[i])
            tp_['cdr1_heavy']=process(cdr1hs[i])
            tp_['cdr2_heavy']=process(cdr2hs[i])
            tp_['cdr3_heavy']=process(cdr3hs[i])
            tp_['sequence_light']=process(seqls[i])
            tp_['cdr1_light']=process(cdr1ls[i])
            tp_['cdr2_light']=process(cdr2ls[i])
            tp_['cdr3_light']=process(cdr3ls[i])
            data_train.append(tp_)

    print(len(data_train))
    json_data=json.dumps(data_train)

    with open('data/train/{}.json'.format(chain), 'w') as f:
        f.write(json_data)
    
    del json_data, data_train
    
    data_test=[]
    for file in tqdm(test_files):
        mask_=(df['file_name']==file)
        
        seqhs=df['sequence_heavy'][mask_].values
        cdr1hs=df['cdr1_heavy'][mask_].values
        cdr2hs=df['cdr2_heavy'][mask_].values
        cdr3hs=df['cdr3_heavy'][mask_].values
        seqls=df['sequence_light'][mask_].values
        cdr1ls=df['cdr1_light'][mask_].values
        cdr2ls=df['cdr2_light'][mask_][mask_].values
        cdr3ls=df['cdr3_light'][mask_].values

        for i in range(len(seqhs)):
            tp_=dict()
            tp_['sequence_heavy']=process(seqhs[i])
            tp_['cdr1_heavy']=process(cdr1hs[i])
            tp_['cdr2_heavy']=process(cdr2hs[i])
            tp_['cdr3_heavy']=process(cdr3hs[i])
            tp_['sequence_light']=process(seqls[i])
            tp_['cdr1_light']=process(cdr1ls[i])
            tp_['cdr2_light']=process(cdr2ls[i])
            tp_['cdr3_light']=process(cdr3ls[i])
            data_test.append(tp_)
    
    print(len(data_test))
    json_data=json.dumps(data_test)

    with open('data/test/{}.json'.format(chain), 'w') as f:
        f.write(json_data)
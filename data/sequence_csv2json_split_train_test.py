import os
import json
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

# split train and test set based on files

if 1:
    chain='heavy'
    path='antibody_data/processed_dataset/sequence/unpaired_heavy_rep_sequences_all_0.3_dedup.csv'

    test_prop=0.1
    df=pd.read_csv(path) #['sequence', 'germline', 'cdr1', 'cdr2', 'cdr3', 'species', 'file_name']
    files=df['file_name'].unique()
    files.sort()

    num_all=len(df)
    print('file all: {}, data all: {}'.format(len(files),num_all))

    test_num=num_all*test_prop
    test_files=[]
    file_ids=np.arange(len(files))
    random.shuffle(file_ids)

    idx=0
    num=0
    while num < test_num:
        add_ = (df['file_name']==files[file_ids[idx]]).sum()
        if num + add_ > test_num * 1.1:
            idx+=1
            continue
        elif num + add_ > test_num * 0.95:
            test_files.append(files[file_ids[idx]])
            num+=add_
            break
        else:
            test_files.append(files[file_ids[idx]])
            num+=add_
            idx+=1

    train_files=[]
    for file in files:
        if file not in test_files:
            train_files.append(file)

    np.save('data/{}_train.npy'.format(chain),train_files)
    np.save('data/{}_test.npy'.format(chain),test_files)



def process(data):
    if pd.isna(data):
        return ''
    else:
        return data

if 1:
    chain='heavy'
    train_files=np.load('data/{}_train.npy'.format(chain),allow_pickle=True)
    test_files=np.load('data/{}_test.npy'.format(chain),allow_pickle=True)
    
    path='antibody_data/processed_dataset/sequence/unpaired_heavy_rep_sequences_all_0.3_dedup.csv'

    df=pd.read_csv(path) #['sequence', 'germline', 'cdr1', 'cdr2', 'cdr3', 'species', 'file_name']
    file_names=df['file_name'].values
    
    data_train=[]
    for file in tqdm(train_files):
        mask_=(file_names==file)
        
        seqs=df['sequence'][mask_].values
        cdr1s=df['cdr1'][mask_].values
        cdr2s=df['cdr2'][mask_].values
        cdr3s=df['cdr3'][mask_].values

        for i in range(len(seqs)):
            tp_=dict()
            tp_['sequence']=process(seqs[i])
            tp_['cdr1']=process(cdr1s[i])
            tp_['cdr2']=process(cdr2s[i])
            tp_['cdr3']=process(cdr3s[i])
            data_train.append(tp_)
    print(len(data_train))
    json_data=json.dumps(data_train)

    with open('data/train/{}.json'.format(chain), 'w') as f:
        f.write(json_data)
    
    del json_data, data_train
    
    data_test=[]
    for file in tqdm(test_files):
        mask_=(df['file_name']==file)
        
        seqs=df['sequence'][mask_].values
        cdr1s=df['cdr1'][mask_].values
        cdr2s=df['cdr2'][mask_].values
        cdr3s=df['cdr3'][mask_].values

        
        for i in range(len(seqs)):
            tp_=dict()
            tp_['sequence']=process(seqs[i])
            tp_['cdr1']=process(cdr1s[i])
            tp_['cdr2']=process(cdr2s[i])
            tp_['cdr3']=process(cdr3s[i])
            data_test.append(tp_)
    print(len(data_test))
    json_data=json.dumps(data_test)

    with open('data/test/{}.json'.format(chain), 'w') as f:
        f.write(json_data)









if 1:
    chain='paired'
    path='antibody_data/processed_dataset/sequence/paired_rep_sequences_all_0.7.csv'
    test_prop=0.1

    df=pd.read_csv(path)
    files=df['file_name'].unique() #sequence_heavy,germline_heavy,cdr1_heavy,cdr2_heavy,cdr3_heavy,sequence_light,germline_light,cdr1_light,cdr2_light,cdr3_light,species,file_name
    files.sort()

    num_all=len(df)
    print('file all: {}, data all: {}'.format(len(files),num_all))

    test_num=num_all*test_prop
    test_files=[]
    file_ids=np.arange(len(files))
    random.shuffle(file_ids)

    idx=0
    num=0
    while num < test_num:
        add_ = (df['file_name']==files[file_ids[idx]]).sum()
        if num + add_ > test_num * 1.1:
            idx+=1
            continue
        elif num + add_ > test_num * 0.95:
            test_files.append(files[file_ids[idx]])
            num+=add_
            break
        else:
            test_files.append(files[file_ids[idx]])
            num+=add_
            idx+=1

    train_files=[]
    for file in files:
        if file not in test_files:
            train_files.append(file)

    np.save('data/{}_train.npy'.format(chain),train_files)
    np.save('data/{}_test.npy'.format(chain),test_files)



def process(data):
    if pd.isna(data):
        return ''
    else:
        return data

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
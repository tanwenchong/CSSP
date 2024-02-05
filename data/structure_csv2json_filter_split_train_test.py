import os
import json
import pandas as pd

from tqdm import tqdm
import random

# csv2json with filter and train test split

def process(data):
    if pd.isna(data):
        return ''
    else:
        return data

def csv2json(idx, chain, train_prop, thres):
    print('thres:',thres[chain])
    if chain=='paired':
        df=pd.read_csv('antibody_data/processed_dataset/struct/{}_3di_seq_{}.csv'.format(chain,idx))
        thres_df=pd.read_csv('antibody_data/structure_data/cal_rmsd/{}{}_rmsd.csv'.format(chain,idx))
        thres_file=thres_df['file'].values
        thres_rmsd=thres_df['rmsd'].values
        del thres_df

        seqhs=df['sequence_heavy'].values
        cdr1hs=df['cdr1_heavy'].values
        cdr2hs=df['cdr2_heavy'].values
        cdr3hs=df['cdr3_heavy'].values
        seqls=df['sequence_light'].values
        cdr1ls=df['cdr1_light'].values
        cdr2ls=df['cdr2_light'].values
        cdr3ls=df['cdr3_light'].values
        pdbs=df['pdb_file'].values

        data_train=[]
        data_test=[]

        for i in tqdm(range(len(df))):
            pdb_=pdbs[i]
            rmsd_=thres_rmsd[thres_file==pdb_]
            assert len(rmsd_)==1
            rmsd_=rmsd_[0]
            if rmsd_ >= thres[chain]:
                continue
            
            tp_=dict()
            tp_['sequence_heavy']=process(seqhs[i])
            tp_['cdr1_heavy']=process(cdr1hs[i])
            tp_['cdr2_heavy']=process(cdr2hs[i])
            tp_['cdr3_heavy']=process(cdr3hs[i])
            tp_['sequence_light']=process(seqls[i])
            tp_['cdr1_light']=process(cdr1ls[i])
            tp_['cdr2_light']=process(cdr2ls[i])
            tp_['cdr3_light']=process(cdr3ls[i])
            tp_['pdb_file']=process(pdbs[i])
            if random.random()<train_prop:
                data_train.append(tp_)
            else:
                data_test.append(tp_)
        
        json_train=json.dumps(data_train)
        with open('data/train/{}{}.json'.format(chain,idx), 'w') as f:
            f.write(json_train)

        json_test=json.dumps(data_test)
        with open('data/test/{}{}.json'.format(chain,idx), 'w') as f:
            f.write(json_test)
    
    else:
        df=pd.read_csv('antibody_data/processed_dataset/struct/{}_3di_seq_{}.csv'.format(chain,idx))
        thres_df=pd.read_csv('antibody_data/structure_data/cal_rmsd/{}{}_rmsd.csv'.format(chain,idx))
        thres_file=thres_df['file'].values
        thres_rmsd=thres_df['rmsd'].values
        del thres_df
        
        seqs=df['sequence'].values
        cdr1s=df['cdr1'].values
        cdr2s=df['cdr2'].values
        cdr3s=df['cdr3'].values
        pdbs=df['pdb_file'].values

        data_train=[]
        data_test=[]

        for i in tqdm(range(len(df))):
            pdb_=pdbs[i]
            rmsd_=thres_rmsd[thres_file==pdb_]
            assert len(rmsd_)==1
            rmsd_=rmsd_[0]
            if rmsd_ >= thres[chain]:
                continue
            
            tp_=dict()
            tp_['sequence']=process(seqs[i])
            tp_['cdr1']=process(cdr1s[i])
            tp_['cdr2']=process(cdr2s[i])
            tp_['cdr3']=process(cdr3s[i])
            tp_['pdb_file']=process(pdbs[i])
            
            if random.random()<train_prop:
                data_train.append(tp_)
            else:
                data_test.append(tp_)
        
        json_train=json.dumps(data_train)
        with open('data/train/{}{}.json'.format(chain,idx), 'w') as f:
            f.write(json_train)

        json_test=json.dumps(data_test)
        with open('data/test/{}{}.json'.format(chain,idx), 'w') as f:
            f.write(json_test)


chain_types=['paired','heavy','light']
train_prop=0.9
filter_thres={
    'heavy':1.9,
    'light':1.9,
    'paired':1.5
} # <thres

for chain in chain_types:
    for i in range(5):
        csv2json(i, chain, train_prop = train_prop, thres = filter_thres)
import os
import numpy as np
import pandas as pd

from tqdm import tqdm

idx_=0

df_all=pd.read_csv('antibody_data/processed_dataset/struct/paired_struct_{}.csv'.format(idx_))
df_3di=pd.read_csv('antibody_data/processed_dataset/struct/paired_3di_seq_{}.csv'.format(idx_))

for i in tqdm(range(len(df_3di))):
    seqh_3di=df_3di['sequence_heavy'][i]
    cdr1h_3di=df_3di['cdr1_heavy'][i]
    cdr2h_3di=df_3di['cdr2_heavy'][i]
    cdr3h_3di=df_3di['cdr3_heavy'][i]
    
    seql_3di=df_3di['sequence_light'][i]
    cdr1l_3di=df_3di['cdr1_light'][i]
    cdr2l_3di=df_3di['cdr2_light'][i]
    cdr3l_3di=df_3di['cdr3_light'][i]
    
    idxh=np.where((df_all['3di_heavy']==seqh_3di).values)[0]
    idxl=np.where((df_all['3di_light']==seql_3di).values)[0]
    
    for idx_ in idxh:
        if idx_ in idxl:
            idx=idx_
            break
    
    seqh_aa=df_all['sequence_heavy'][idx]
    cdr1h_aa=df_all['cdr1_heavy'][idx]
    cdr2h_aa=df_all['cdr2_heavy'][idx]
    cdr3h_aa=df_all['cdr3_heavy'][idx]
    
    seql_aa=df_all['sequence_light'][idx]
    cdr1l_aa=df_all['cdr1_light'][idx]
    cdr2l_aa=df_all['cdr2_light'][idx]
    cdr3l_aa=df_all['cdr3_light'][idx]
    
    assert len(seqh_3di)==len(seqh_aa)
    assert len(seql_3di)==len(seql_aa)
    
    
    if(pd.isna(cdr1h_3di)):
        assert pd.isna(cdr1h_aa)
    else:
        assert len(cdr1h_3di)==len(cdr1h_aa)
        try:
            assert seqh_3di.find(cdr1h_3di)==seqh_aa.find(cdr1h_aa)
        except:
            print('cdr1h',i,cdr1h_3di)
        
    if(pd.isna(cdr2h_3di)):
        assert pd.isna(cdr2h_aa)
    else:
        assert len(cdr2h_3di)==len(cdr2h_aa)
        try:
            assert seqh_3di.find(cdr2h_3di)==seqh_aa.find(cdr2h_aa)
        except:
            print('cdr2h',i,cdr2h_3di)
        
    if(pd.isna(cdr3h_3di)):
        assert pd.isna(cdr3h_aa)
    else:
        assert len(cdr3h_3di)==len(cdr3h_aa)
        try:
            assert seqh_3di.find(cdr3h_3di)==seqh_aa.find(cdr3h_aa)
        except:
            print('cdr3h',i,cdr3h_3di)
            
            
            
    if(pd.isna(cdr1l_3di)):
        assert pd.isna(cdr1l_aa)
    else:
        assert len(cdr1l_3di)==len(cdr1l_aa)
        try:
            assert seql_3di.find(cdr1l_3di)==seql_aa.find(cdr1l_aa)
        except:
            print('cdr1l',i,cdr1l_3di)
        
    if(pd.isna(cdr2l_3di)):
        assert pd.isna(cdr2l_aa)
    else:
        assert len(cdr2l_3di)==len(cdr2l_aa)
        try:
            assert seql_3di.find(cdr2l_3di)==seql_aa.find(cdr2l_aa)
        except:
            print('cdr2l',i,cdr2l_3di)
        
    if(pd.isna(cdr3l_3di)):
        assert pd.isna(cdr3l_aa)
    else:
        assert len(cdr3l_3di)==len(cdr3l_aa)
        try:
            assert seql_3di.find(cdr3l_3di)==seql_aa.find(cdr3l_aa)
        except:
            print('cdr3l',i,cdr3l_3di)
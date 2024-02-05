import os
import numpy as np
import pandas as pd

from tqdm import tqdm

chain='heavy'
idx_=0

df_3di=pd.read_csv('/code/data_process/struct_data_process/{}_3di_seq_{}.csv'.format(chain,idx_))

df_all=pd.read_csv('antibody_data/processed_dataset/struct/heavy_struct_{}.csv'.format(idx_))

def get_all_index(seq,sub):
    ret=[]
    tp=seq
    while True:
        idx=tp.find(sub)
        if idx < 0:
            break
        else:
            ret.append(idx)
            tp=tp[:idx]+'#'+tp[idx+1:]
    return ret


for i in tqdm(range(len(df_3di))):
    seq_3di=df_3di['sequence'][i]
    cdr1_3di=df_3di['cdr1'][i]
    cdr2_3di=df_3di['cdr2'][i]
    cdr3_3di=df_3di['cdr3'][i]
    
    idx=np.where((df_all['3di']==seq_3di).values)[0][0]
    seq_aa=df_all['sequence'][idx]
    cdr1_aa=df_all['cdr1'][idx]
    cdr2_aa=df_all['cdr2'][idx]
    cdr3_aa=df_all['cdr3'][idx]
    
    assert len(seq_3di)==len(seq_aa)
    
    if(pd.isna(cdr1_3di)):
        assert pd.isna(cdr1_aa)
    else:
        assert len(cdr1_3di)==len(cdr1_aa)
        ret=get_all_index(seq_3di,cdr1_3di)
        assert seq_aa.find(cdr1_aa) in ret
        
    if(pd.isna(cdr2_3di)):
        assert pd.isna(cdr2_aa)
    else:
        assert len(cdr2_3di)==len(cdr2_aa)
        ret=get_all_index(seq_3di,cdr2_3di)
        assert seq_aa.find(cdr2_aa) in ret
        
    if(pd.isna(cdr3_3di)):
        assert pd.isna(cdr3_aa)
    else:
        assert len(cdr3_3di)==len(cdr3_aa)
        ret=get_all_index(seq_3di,cdr3_3di)
        assert seq_aa.find(cdr3_aa) in ret

import os
import numpy as np
import pandas as pd

chain='light' # 'heavy'
# tsv from mmseqs
cluster_file = 'clusterRes_light_0.6_cluster.tsv'
# path to processed data
root= 'antibody_data/merge_sequence/{}/'.format(chain)

df=pd.read_csv(os.path.join(root, cluster_file),sep='\t',names=['rep','seq'])
reps=df['rep'].unique()
print(len(reps))

if 1:
    df=pd.read_csv(os.path.join(root,'unpaired_{}.csv'.format(chain)))
    print(df.shape)
    df_=df.loc[reps]
    print(df_.shape)
    print(df_.head())
    print(df_.iloc[0])
    
    dups=df_['sequence'].duplicated().values
    print(dups)
    print(dups.sum())
    df_=df_[~dups]
    print(df_.shape)
    df_.to_csv(os.path.join(root,'unpaired_{}_rep_sequences_all_dedup.csv'.format(chain)),index=False)


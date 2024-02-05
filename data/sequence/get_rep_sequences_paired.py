import os
import numpy as np
import pandas as pd

# path to processed data
root= 'antibody_data/merge_sequence/paired/'
# tsv from mmseqs
cluster_file = 'clusterRes_paired_0.7_cluster.tsv'

df=pd.read_csv(os.path.join(root, cluster_file),sep='\t',names=['rep','seq'])
reps=df['rep'].unique()
print(len(reps))

if 1:
    df=pd.read_csv(os.path.join(root,'paired.csv'))
    print(df.shape)
    df_=df.loc[reps]
    print(df_.shape)
    print(df_.head())
    print(df_.iloc[0])
    
    df_.to_csv(os.path.join(root,'paired_rep_sequences_all_dedup.csv'),index=False)

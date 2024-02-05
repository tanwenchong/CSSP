import numpy as np
import pandas as pd


# change file path accordingly

# light
if 1:
    df=pd.read_csv('antibody_data/merge_sequence/light/unpaired_light_rep_sequences_all_0.4_dedup.csv')
    sequences=df['sequence'].values

    len_=len(sequences)

    per_num=(len_//5)+1

    for i in range(0, len_, per_num):
        seqs_=sequences[i:i+per_num]
        print(seqs_.shape)
        print(seqs_[0])
        np.save('antibody_data/structure_data/light/seq_split/sequences_{}.npy'.format(i//per_num),seqs_)

# paired
if 1:
    df=pd.read_csv('antibody_data/merge_sequence/paired/paired_rep_sequences_all_0.7.csv')
    hseqs=df['sequence_heavy'].values
    lseqs=df['sequence_light'].values
    
    len_=len(df)
    
    per_num=(len_//5)+1
    
    for i in range(0, len_, per_num):
        hseqs_=hseqs[i:i+per_num]
        lseqs_=lseqs[i:i+per_num]
        print(hseqs_.shape)
        print(hseqs_[0])
        np.save('antibody_data/structure_data/paired/seq_split/sequences_heavy_{}.npy'.format(i//per_num),hseqs_)
        print(lseqs_.shape)
        print(lseqs_[0])
        np.save('antibody_data/structure_data/paired/seq_split/sequences_light_{}.npy'.format(i//per_num),lseqs_)


# heavy
if 1:
    df=pd.read_csv('antibody_data/merge_sequence/heavy/unpaired_heavy_rep_sequences_all_0.3_dedup.csv')
    sequences=df['sequence'].values

    len_=len(sequences)

    per_num=(len_//5)+1

    for i in range(0, len_, per_num):
        seqs_=sequences[i:i+per_num]
        print(seqs_.shape)
        print(seqs_[0])
        np.save('antibody_data/structure_data/heavy/seq_split/sequences_{}.npy'.format(i//per_num),seqs_)


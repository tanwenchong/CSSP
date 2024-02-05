import os
import numpy as np
import pandas as pd

from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1
from Bio import SeqIO

from tqdm import tqdm

# check pdb seq and npy sequence
if 0:
    idx_=0
    seq_npy=np.load('antibody_data/structure_data/light/seq_split/sequences_{}.npy'.format(idx_),allow_pickle=True)
    for i in tqdm(range(len(seq_npy))):
        pdb_path_='antibody_data/structure_data/light/split_predict/{}/output_{}_{}.pdb'.format(idx_,idx_,i)
        parser = PDBParser(PERMISSIVE=1)
        residues = parser.get_structure('tp', pdb_path_)[0]['A'].get_list()
        seq_pdb=''
        for res in residues:
            if res.get_resname() in ['GAP']:
                seq_pdb+='X'
                continue
            seq_pdb+=protein_letters_3to1[res.get_resname()]
        assert seq_pdb==seq_npy[i]


def get_cdr_struct(seq_aa, cdr1_aa, cdr2_aa, cdr3_aa, seq_3di):
    assert len(seq_aa)==len(seq_3di)
    if pd.isna(cdr1_aa):
        cdr1_3di=None
    else:
        idx_=seq_aa.find(cdr1_aa)
        cdr1_3di=seq_3di[idx_:idx_+len(cdr1_aa)]
    if pd.isna(cdr2_aa):
        cdr2_3di=None
    else:
        idx_=seq_aa.find(cdr2_aa)
        cdr2_3di=seq_3di[idx_:idx_+len(cdr2_aa)]
    if pd.isna(cdr3_aa):
        cdr3_3di=None
    else:
        idx_=seq_aa.find(cdr3_aa)
        cdr3_3di=seq_3di[idx_:idx_+len(cdr3_aa)]
    return seq_3di, cdr1_3di, cdr2_3di, cdr3_3di


chain='light'

data_all=pd.read_csv('antibody_data/merge_sequence/light/unpaired_light_rep_sequences_all_0.4_dedup.csv')

# 'sequence', 'germline', 'cdr1', 'cdr2', 'cdr3', 'species', 'file_name'
columns=data_all.columns
sequences=data_all['sequence'].values
cdr1s=data_all['cdr1'].values
cdr2s=data_all['cdr2'].values
cdr3s=data_all['cdr3'].values
species=data_all['species'].values
file_names=data_all['file_name'].values
del data_all

for idx_ in range(5):
    result_all=[]
    
    seq_npy=np.load('antibody_data/structure_data/light/seq_split/sequences_{}.npy'.format(idx_),allow_pickle=True)#[:1000]
    f_fasta=SeqIO.parse('antibody_data/structure_data/3di/light/DB_light_{}_ss.fasta'.format(idx_),'fasta')

    fasta_dict=dict()
    for item in f_fasta:
        fasta_dict[item.name]=item.seq

    for i in tqdm(range(len(seq_npy))):
        index_=np.where(sequences==seq_npy[i])[0][0]
        #print(index_)
        seq_, cdr1_, cdr2_, cdr3_ = get_cdr_struct(sequences[index_], cdr1s[index_], cdr2s[index_], cdr3s[index_], fasta_dict['output_{}_{}.pdb'.format(idx_, i)])
        result_all.append([seq_, cdr1_, cdr2_, cdr3_, species[index_], file_names[index_], 'output_{}_{}.pdb'.format(idx_, i)])

    pd.DataFrame(result_all,columns=['sequence', 'cdr1', 'cdr2', 'cdr3', 'species', 'file_name', 'pdb_file']).to_csv('antibody_data/processed_dataset/struct/light_3di_seq_{}.csv'.format(idx_), index=False)


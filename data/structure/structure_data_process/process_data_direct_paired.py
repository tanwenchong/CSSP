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
    seqh_npy=np.load('antibody_data/structure_data/paired/seq_split/sequences_heavy_{}.npy'.format(idx_), allow_pickle=True)
    seql_npy=np.load('antibody_data/structure_data/paired/seq_split/sequences_light_{}.npy'.format(idx_), allow_pickle=True)
    for i in tqdm(range(len(seqh_npy))):
        pdb_path_='antibody_data/structure_data/paired/split_predict/{}/output_{}_{}.pdb'.format(idx_,idx_,i)
        parser = PDBParser(PERMISSIVE=1)
        structure=parser.get_structure('tp', pdb_path_)[0]
        residues = structure['H'].get_list()
        seq_pdb=''
        for res in residues:
            if res.get_resname() in ['GAP']:
                seq_pdb+='X'
                continue
            seq_pdb+=protein_letters_3to1[res.get_resname()]
        assert seq_pdb==seqh_npy[i]
        
        residues = structure['L'].get_list()
        seq_pdb=''
        for res in residues:
            if res.get_resname() in ['GAP']:
                seq_pdb+='X'
                continue
            seq_pdb+=protein_letters_3to1[res.get_resname()]
        assert seq_pdb==seql_npy[i]


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



data_all=pd.read_csv('antibody_data/merge_sequence/paired/paired_rep_sequences_all_0.7.csv')


# 'sequence_heavy', 'cdr1_heavy', 'cdr2_heavy', 'cdr3_heavy', 'sequence_light', 'cdr1_light', 'cdr2_light', 'cdr3_light', 'species', 'file_name'
columns=data_all.columns

sequences_h=data_all['sequence_heavy'].values
cdr1s_h=data_all['cdr1_heavy'].values
cdr2s_h=data_all['cdr2_heavy'].values
cdr3s_h=data_all['cdr3_heavy'].values
sequences_l=data_all['sequence_light'].values
cdr1s_l=data_all['cdr1_light'].values
cdr2s_l=data_all['cdr2_light'].values
cdr3s_l=data_all['cdr3_light'].values
species=data_all['species'].values
file_names=data_all['file_name'].values
del data_all

for idx_ in range(5):
    result_all=[]
    seqh_npy=np.load('antibody_data/structure_data/paired/seq_split/sequences_heavy_{}.npy'.format(idx_), allow_pickle=True)[:1000]

    seql_npy=np.load('antibody_data/structure_data/paired/seq_split/sequences_light_{}.npy'.format(idx_), allow_pickle=True)[:1000]

    f_fasta=SeqIO.parse('antibody_data/structure_data/3di/paired/DB_paired_{}_ss.fasta'.format(idx_),'fasta')

    fasta_dict=dict()
    fasta_dict['H']=dict()
    fasta_dict['L']=dict()

    for item in f_fasta:
        pdb_,chain_=item.name[:-2],item.name[-1]
        fasta_dict[chain_][pdb_]=item.seq

    for i in tqdm(range(len(seqh_npy))):
        indexh_=np.where(sequences_h==seqh_npy[i])[0]
        indexl_=np.where(sequences_l==seql_npy[i])[0]

        for idx_h in indexh_:
            if idx_h in indexl_:
                index_=idx_h
                break

        #print(index_)
        seqh_,cdr1h_,cdr2h_,cdr3h_=get_cdr_struct(sequences_h[index_], cdr1s_h[index_], cdr2s_h[index_], cdr3s_h[index_], fasta_dict['H']['output_{}_{}.pdb'.format(idx_, i)])
        seql_,cdr1l_,cdr2l_,cdr3l_=get_cdr_struct(sequences_l[index_], cdr1s_l[index_], cdr2s_l[index_], cdr3s_l[index_], fasta_dict['L']['output_{}_{}.pdb'.format(idx_, i)])
        result_all.append([seqh_, cdr1h_, cdr2h_, cdr3h_, 
                           seql_, cdr1l_, cdr2l_, cdr3l_, 
                           species[index_], file_names[index_], 
                           'output_{}_{}.pdb'.format(idx_, i)])

    pd.DataFrame(result_all,columns=[
        'sequence_heavy', 'cdr1_heavy', 'cdr2_heavy', 'cdr3_heavy',
        'sequence_light', 'cdr1_light', 'cdr2_light', 'cdr3_light',
        'species', 'file_name', 'pdb_file'
    ]).to_csv('antibody_data/processed_dataset/struct/paired_3di_seq_{}.csv'.format(idx_),index=False)

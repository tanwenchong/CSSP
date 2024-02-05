import os
import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1

from tqdm import tqdm

idx_=0

f_fasta=SeqIO.parse('antibody_data/structure_data/3di/paired/DB_paired_{}_ss.fasta'.format(idx_),'fasta')

fasta_dict=dict()
fasta_dict['H']=dict()
fasta_dict['L']=dict()

for item in f_fasta:
    pdb_,chain_=item.name[:-2],item.name[-1]
    fasta_dict[chain_][pdb_]=item.seq



df=pd.read_csv('antibody_data/processed_dataset/struct/paired_struct_{}.csv'.format(idx_))

for i in tqdm(range(len(df))):
    df_=df.iloc[i]
    
    assert pd.isna(df_['cdr1_heavy']) or (df_['cdr1_heavy'] in df_['sequence_heavy'])
    assert pd.isna(df_['cdr2_heavy']) or (df_['cdr2_heavy'] in df_['sequence_heavy'])
    assert pd.isna(df_['cdr3_heavy']) or (df_['cdr3_heavy'] in df_['sequence_heavy'])
    assert pd.isna(df_['cdr1_light']) or (df_['cdr1_light'] in df_['sequence_light'])
    assert pd.isna(df_['cdr2_light']) or (df_['cdr2_light'] in df_['sequence_light'])
    assert pd.isna(df_['cdr3_light']) or (df_['cdr3_light'] in df_['sequence_light'])
    
    assert fasta_dict['H'][df_['pdb_file']]==df_['3di_heavy']
    assert fasta_dict['L'][df_['pdb_file']]==df_['3di_light']
    
    pdb_path_='antibody_data/structure_data/paired/split_predict/{}/{}'.format(idx_,df_['pdb_file'])
    parser = PDBParser(PERMISSIVE=1)
    structure=parser.get_structure('tp', pdb_path_)[0]
    residues = structure['H'].get_list()
    seq_pdb=''
    for res in residues:
        if res.get_resname() in ['GAP']:
            seq_pdb+='X'
            continue
        seq_pdb+=protein_letters_3to1[res.get_resname()]
    assert seq_pdb==df_['sequence_heavy']

    residues = structure['L'].get_list()
    seq_pdb=''
    for res in residues:
        if res.get_resname() in ['GAP']:
            seq_pdb+='X'
            continue
        seq_pdb+=protein_letters_3to1[res.get_resname()]

    assert seq_pdb==df_['sequence_light']

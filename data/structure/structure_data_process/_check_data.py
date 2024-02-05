import os
import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1

from tqdm import tqdm

chain='heavy'
idx_=0

f_fasta=SeqIO.parse('antibody_data/structure_data/3di/{}/DB_{}_{}_ss.fasta'.format(chain,chain,idx_),'fasta')

fasta_dict=dict()
for item in f_fasta:
    fasta_dict[item.name]=item.seq


df=pd.read_csv('/code/data_process/struct_data_process/{}_struct_{}.csv'.format(chain,idx_))

for i in tqdm(range(len(df))):
    df_=df.iloc[i]
    
    assert pd.isna(df_['cdr1']) or (df_['cdr1'] in df_['sequence'])
    assert pd.isna(df_['cdr2']) or (df_['cdr2'] in df_['sequence'])
    assert pd.isna(df_['cdr3']) or (df_['cdr3'] in df_['sequence'])
    
    assert fasta_dict[df_['pdb_file']]==df_['3di']
    
    pdb_path_='antibody_data/structure_data/{}/split_predict/{}/{}'.format(chain,idx_,df_['pdb_file'])
    parser = PDBParser(PERMISSIVE=1)
    residues = parser.get_structure('tp', pdb_path_)[0]['A'].get_list()
    seq_pdb=''
    for res in residues:
        if res.get_resname() in ['GAP']:
            seq_pdb+='X'
            continue
        seq_pdb+=protein_letters_3to1[res.get_resname()]
        
    assert seq_pdb==df_['sequence']
    
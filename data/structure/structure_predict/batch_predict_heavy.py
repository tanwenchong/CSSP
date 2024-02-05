import os
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from einops import rearrange

from antiberty import AntiBERTyRunner

from igfold.model.IgFold import IgFold
from igfold.utils.abnumber_ import truncate_seq, renumber_pdb
from igfold.utils.folding import process_template
from igfold.utils.pdb import get_atom_coords, save_PDB, write_pdb_bfactor, cdr_indices
from igfold.model.interface import IgFoldInput

import re

file_id=0 # 0-4

RENUMBER=False
batch_size=1
file_to_predict='antibody_data/structure_data/heavy/seq_split/sequences_{}.npy'.format(file_id)


tp_fasta_file='temp/temp_heavy_{}.fasta'.format(file_id)
device=torch.device('cuda')
chain_name='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

assert len(chain_name)>batch_size

# Noted that we only use igfold_1 here for efficiency 
# as igfold_1 has the best and most stable performance among all the ckpts empirically.

ckpt_file='igfold/trained_models/IgFold/igfold_1.ckpt'
model=IgFold.load_from_checkpoint(ckpt_file).eval().to(device)
antiberty = AntiBERTyRunner()
antiberty.model.eval()
antiberty.model.to(device)

sequence_to_predict=np.load(file_to_predict,allow_pickle=True).tolist()

# truncate
#sequence_to_predict=pd.Series(sequence_to_predict)
#sequence_to_predict=sequence_to_predict.apply(truncate_seq)
#print(sequence_to_predict.shape)
#sequence_to_predict=sequence_to_predict.values.tolist()
#count=0
#for i in tqdm(range(len(sequence_to_predict))):
    #print(sequence_to_predict[i])
#    try:
#        sequence_to_predict[i]=truncate_seq(sequence_to_predict[i])
#    except:
#        count+=1
#        print(count)


sequence_all=len(sequence_to_predict)

for i in tqdm(range(0,sequence_all,batch_size)):
    pdb_file='antibody_data/structure_data/heavy/split_predict/{}/output_{}_{}.pdb'.format(file_id, 
                                                                                           file_id, 
                                                                                           i//batch_size)
    sequence_=sequence_to_predict[i:i+batch_size]
    for i in range(len(sequence_)):
        sequence_[i]=re.sub(r'[\*\-]','',sequence_[i])
    
    chain_keys=[]
    
    with open(tp_fasta_file, "w") as f:
        for num_i, seq in enumerate(sequence_):
            f.write(">{}\n{}\n".format(
                chain_name[num_i],
                seq,
            ))
            chain_keys.append(chain_name[num_i])
            
    assert len(sequence_)==len(chain_keys)

    embeddings, attentions = antiberty.embed(
        sequence_,
        return_attention=True,
    )
    embeddings = [e[1:-1].unsqueeze(0) for e in embeddings]
    attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0) for a in attentions]

    temp_coords, temp_mask = process_template(
        None, # template_pdb
        tp_fasta_file,
        ignore_cdrs= None,
        ignore_chain=None,
    )
    model_in = IgFoldInput(
        embeddings=embeddings,
        attentions=attentions,
        template_coords=temp_coords,
        template_mask=temp_mask,
        return_embeddings=True,
    )

    with torch.no_grad():
        model_out = model(model_in)
        model_out = model.gradient_refine(model_in, model_out)

    
    prmsd = rearrange(
        model_out.prmsd,
        "b (l a) -> b l a",
        a=4,
    )
    model_out.prmsd = prmsd
    #print('prmsd shape',prmsd.shape)

    coords = model_out.coords.squeeze(0).detach()
    res_rmsd = prmsd.square().mean(dim=-1).sqrt().squeeze(0)
    #print('res_rmsd shape',res_rmsd.shape)

    full_seq = "".join(sequence_)
    delims = np.cumsum([len(s) for s in sequence_]).tolist()
    
    full_seq=re.sub('X','-',full_seq)
    
    pdb_string = save_PDB(
        pdb_file,
        coords,
        full_seq,
        chains=chain_keys,
        atoms=['N', 'CA', 'C', 'CB', 'O'],
        error=res_rmsd,
        delim=delims,
        write_pdb=True,
    )

    if RENUMBER:
        renumber_pdb(pdb_file, pdb_file)

    write_pdb_bfactor(pdb_file, pdb_file, bfactor=res_rmsd)


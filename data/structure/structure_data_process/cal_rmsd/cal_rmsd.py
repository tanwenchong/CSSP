import os
import numpy as np
import pandas as pd

import Bio.PDB.PDBParser as PDBParser
from tqdm import tqdm

parser=PDBParser(PERMISSIVE=1)

def get_rmsd(path):
    data=parser.get_structure('test',path)
    res_all=list(data.get_residues())
    rmsds=[]
    for i in range(len(res_all)):
        res=res_all[i]
        rmsds.append(list(res.get_atoms())[0].get_bfactor())
    return np.sqrt(np.square(rmsds).mean())

for chain in ['heavy','light','paired']:
    for idx in range(5):
        print(chain,idx)
        data_all=[]
        root_='antibody_data/structure_data/{}/split_predict/{}/'.format(chain,idx)
        files=os.listdir(root_)
        for file in tqdm(files):
            if file.endswith('.pdb'):
                data_all.append([file, get_rmsd(os.path.join(root_,file)), chain, idx])

        data_all=pd.DataFrame(data_all,columns=['file','rmsd','chain','idx'])
        #print(data_all.shape)
        data_all.to_csv('{}{}_rmsd.csv'.format(chain,idx),index=False)
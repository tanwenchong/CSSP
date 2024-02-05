import pandas as pd
import json
from tqdm import tqdm

import random

if 0:
    chain='light'

    for i in range(5):
        df=pd.read_csv('antibody_data/processed_dataset/struct/{}_3di_seq_{}.csv'.format(chain,i))
        seqs=df['sequence'].values
        species=df['species'].values
        files=df['file_name'].values
        pdbs=df['pdb_file'].values

        map_dict=dict()
        for j in range(len(df)):
            map_dict[pdbs[j]]=(seqs[j],species[j],files[j])

        json_origin=json.load(open('../data/test/{}{}.json'.format(chain,i),'r'))
        for j in tqdm(range(len(json_origin))):
            tp_=map_dict[json_origin[j]['pdb_file']]
            assert json_origin[j]['sequence']==tp_[0]
            json_origin[j]['species']=tp_[1]
            json_origin[j]['file_name']=tp_[2]

        json_origin=json.dumps(json_origin)

        with open('../data/dedup/{}{}.json'.format(chain,i),'w') as f:
            f.write(json_origin)


if 0:
    chain='paired'

    for i in range(5):
        df=pd.read_csv('antibody_data/processed_dataset/struct/{}_3di_seq_{}.csv'.format(chain,i))
        seqsh=df['sequence_heavy'].values
        seqsl=df['sequence_light'].values
        species=df['species'].values
        files=df['file_name'].values
        pdbs=df['pdb_file'].values

        map_dict=dict()
        for j in range(len(df)):
            map_dict[pdbs[j]]=(seqsh[j],seqsl[j],species[j],files[j])

        json_origin=json.load(open('../data/test/{}{}.json'.format(chain,i),'r'))
        for j in tqdm(range(len(json_origin))):
            tp_=map_dict[json_origin[j]['pdb_file']]
            assert json_origin[j]['sequence_heavy']==tp_[0]
            assert json_origin[j]['sequence_light']==tp_[1]
            json_origin[j]['species']=tp_[2]
            json_origin[j]['file_name']=tp_[3]

        json_origin=json.dumps(json_origin)

        with open('../data/dedup/{}{}.json'.format(chain,i),'w') as f:
            f.write(json_origin)
    
# get species num and sample
if 1:
    chain='paired'
    
    #species_count_dict=dict()
    #for i in range(5):
    #    json_origin=json.load(open('../data/dedup/sample/{}{}.json'.format(chain,i),'r'))
    #    for j in range(len(json_origin)):
    #        spe_=json_origin[j]['species']
    #        try:
    #            species_count_dict[spe_]+=1
    #        except:
    #            species_count_dict[spe_]=1
    #print(species_count_dict) # {'human': 132028, 'rabbit': 336, 'mouse_C57BL/6': 1598, 'mouse_BALB/c': 1910, 'camel': 854, 'rat': 269, 'rhesus': 745, 'mouse': 16, 'mouse_Swiss-Webster': 3, 'HIS-Mouse': 36, 'mouse_RAG2-GFP/129Sve': 4}
    

    for i in range(5):
        sampled_json=[]
        
        human_json=[]
        json_origin=json.load(open('../data/dedup/{}{}.json'.format(chain,i),'r'))
        for j in range(len(json_origin)):
            if json_origin[j]['species'] == 'human':
                human_json.append(json_origin[j])
            else:
                sampled_json.append(json_origin[j])
        sampled_json+=random.sample(human_json,2000)
        
        sampled_json=json.dumps(sampled_json)

        with open('../data/dedup/sample/{}{}.json'.format(chain,i),'w') as f:
            f.write(sampled_json)
            
        
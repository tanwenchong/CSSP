'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import re
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model_pretrain import Model

import utils
from torch.utils.data import Dataset

from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1

from dataset.focused_mask import get_focused_span_mask
from models.modeling_bert import BertConfig, BertForMaskedModeling

from tqdm import tqdm

protein_letters_3to1['GAP']='X'

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}

# single chain and paired chain
class inference_dataset(Dataset):
    def __init__(self, json_files, alphabet, max_sl, device):
        
        # pdb_root: 'antibody_data/structure_data/'
        
        self.data_all, self.type_all = self._process_files(json_files)
        
        print('data size',len(self.data_all))
        
        self.parser = PDBParser(PERMISSIVE=1)
        self.alphabet = alphabet
        self.max_sl = max_sl
        self.device=device
        self.map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}
    
    def _process_files(self, json_files):
        data_all=[]
        type_all=[]
        for f in json_files:
            fname=os.path.basename(f)
            chain,idx=re.findall(r'([a-z]+)([0-9]*)\.json',fname)[0]
            assert chain in ['heavy','light','paired']
            data_=json.load(open(f,'r'))
            data_all+=data_
            type_all+=[chain]*len(data_)
        return data_all, type_all

    def __len__(self):
        return len(self.data_all)

    def tokenize(self, seq, max_sl, chain_type, padding=True):
        if type(seq) == tuple:
            seq_id = [self.map_dict['[CLS]']] + [self.map_dict[x] for x in seq[0]] + [self.map_dict['[SEP]']] + [self.map_dict[x] for x in seq[1]] + [self.map_dict['[SEP]']]
            seq_mask = [1]*(1 + len(seq[0]) + 1 + len(seq[1]) + 1)
            seq_type = [0]*(1 + len(seq[0]) + 1) + [1]*(len(seq[1]) + 1)
        else:
            seq_id = [self.map_dict['[CLS]']] + [self.map_dict[x] for x in seq] + [self.map_dict['[SEP]']]
            seq_mask = [1]*(1 + len(seq) + 1)
            if chain_type == 'heavy':
                seq_type = [0]*(1 + len(seq) + 1)
            else:
                seq_type = [1]*(1 + len(seq) + 1)
        
        seq_id = np.array(seq_id)
        seq_mask = np.array(seq_mask)
        seq_type = np.array(seq_type)
        if padding:
            seq_id = np.pad(seq_id.astype(np.int32), (0, max_sl - len(seq_id)),
                            constant_values=self.map_dict['[PAD]'])
            seq_mask = np.pad(seq_mask.astype(np.int32), (0, max_sl - len(seq_mask)), 
                              constant_values=0)
            seq_type = np.pad(seq_type.astype(np.int32), (0, max_sl - len(seq_type)), 
                              constant_values=0)
        
        return torch.tensor(seq_id).long(), torch.tensor(seq_mask).long(), torch.tensor(seq_type).long()

    def preload(self, data, chain):
        if chain=='paired':
            seq_ids, seq_masks, seq_types = self.tokenize((data['sequence_heavy'], data['sequence_light']), self.max_sl, chain)
            
            #print(seq_ids)
            
            return seq_ids, seq_masks, seq_types
        
        else:
            seq_ids, seq_masks, seq_types = self.tokenize(data['sequence'], max_sl = self.max_sl, chain_type = chain)
            
            #print(seq_ids)
            
            return seq_ids, seq_masks, seq_types
    
    def __getitem__(self, index):
        data, chain = self.data_all[index], self.type_all[index]
        #print(chain)
        for key in data.keys():
            if key != 'label':
                data[key] = re.sub(r'[\*\-]','',data[key])

        if chain == 'paired':
            
            if len(data['sequence_heavy']) + len(data['sequence_light']) + 3 > self.max_sl:
                data['sequence_light'] = data['sequence_light'][:self.max_sl - len(data['sequence_heavy']) - 3]
        else:
            if len(data['sequence']) + 2 > self.max_sl:
                data['sequence'] = data['sequence'][:self.max_sl - 2]
        
        #if chain=='paired':
        #    print(chain, len(data['sequence_heavy'])+len(data['sequence_light']))
        #else:
        #    print(chain, len(data['sequence']))

        ret = self.preload(data, chain)

        return ret

class Model(nn.Module):
    def __init__(self,
                 config = None
                 ):
        super().__init__()
        
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.encoder = BertForMaskedModeling(config=bert_config)
        self.max_sl = int(config['max_sl'])

        self.map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}


    def forward(self, input_ids, seq_masks, seq_types):
        '''
        node_features: [B, L, 117]
        edge_features: [B, L, L, 272]
        label_masks: [B, L]
        '''
        
        
        mlm_output = self.encoder(input_ids = input_ids,
                                  attention_mask = seq_masks,
                                  return_dict = True,
                                  labels = None,
                                  vocab_size = len(self.map_dict),
                                  seq_types = seq_types
                                )

        return mlm_output.hidden_states

def main(args, config):
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    alphabet='#####LAGVESIKRDTPNQFYMHCWX'

    #### Dataset ####
    print("Creating dataset")
    datasets = inference_dataset(config['test_file'], alphabet, config['max_sl'], device)

    data_loader = DataLoader(
            datasets,
            batch_size=config['batch_size'],
            num_workers=10,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

    #### Model ####
    print("Creating model")
    model = Model(config=config)
    
    model = model.to(device)
    checkpoint = torch.load(config['ckpt_path'], map_location='cuda')
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    print('load checkpoint from %s'%config['ckpt_path'])
    
    results=[]
    
    
    model.eval()
    with torch.no_grad():
        for i, (input_ids, seq_masks, seq_types) in tqdm(enumerate(data_loader)):
            
            input_ids=input_ids.to(device)
            seq_masks=seq_masks.to(device)
            seq_types=seq_types.to(device)
            last_hidden_state = model(input_ids, seq_masks, seq_types)
            last_hidden_state = last_hidden_state.cpu().data
            
            
            if config['if_pool']:
                assert config['how_pool'] in ['mean', 'max', 'cls']
                if config['how_pool'] == 'mean':
                    # mean pooling
                    seq_masks = seq_masks.cpu().data.unsqueeze(-1) # [B, L, 1]
                    seq_masks = seq_masks[:,1:,:]
                    last_hidden_state = last_hidden_state[:,1:,:]
                    
                    last_hidden_state = last_hidden_state * (seq_masks.repeat(1,1,1024))
                    last_hidden_state = last_hidden_state.sum(1, keepdim=True) / seq_masks.sum(1, keepdim=True) # [B, 1, 1024]
                elif config['how_pool'] == 'max':
                    # max pooling
                    seq_masks = seq_masks.cpu().data.unsqueeze(-1) # [B, L, 1]
                    seq_masks = seq_masks[:,1:,:]
                    last_hidden_state = last_hidden_state[:,1:,:]
                    
                    last_hidden_state = last_hidden_state + (1 - seq_masks.repeat(1,1,1024))*(-100)
                    last_hidden_state = last_hidden_state.max(1, keepdim=True).values # [B, 1, 1024]
                elif config['how_pool'] == 'cls':
                    # cls pooling
                    last_hidden_state = last_hidden_state[:,0,:]
            else:
                # no pooling
                last_hidden_state = last_hidden_state[:,:200,:]
            
            results.append(last_hidden_state.squeeze().numpy())
            
            
    results=np.concatenate(results,axis=0)
    print(results.shape)
    print('save to',config['save_name'])
    np.save(config['save_name'],results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/embedding.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=27, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    main(args, config)
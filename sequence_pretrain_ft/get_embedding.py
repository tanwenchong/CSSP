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
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.model_pretrain import Model

import utils
from torch.utils.data import Dataset

from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1

from dataset.focused_mask import get_focused_span_mask
#from models.modeling_bert import BertConfig, BertForMaskedModeling

from dataset.data import Alphabet
from models.esm2 import ESM2

from tqdm import tqdm

protein_letters_3to1['GAP']='X'

vocabs = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

map_dict = dict()

for i in range(len(vocabs)):
    map_dict[vocabs[i]] = i


def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict



# single chain and paired chain
class inference_dataset(Dataset):
    def __init__(self, json_files, alphabet, max_sl, device, map_dict):
        
        # pdb_root: 'antibody_data/structure_data/'
        
        self.data_all, self.type_all = self._process_files(json_files)
        
        self.parser = PDBParser(PERMISSIVE=1)
        self.alphabet = alphabet
        self.max_sl = max_sl
        self.device=device
        self.map_dict = map_dict
    
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
            seq_id = [self.map_dict['<cls>']] + [self.map_dict[x] for x in seq[0]] + [self.map_dict[x] for x in seq[1]] + [self.map_dict['<eos>']]
            seq_mask = [1]*(1 + len(seq[0]) + len(seq[1]) + 1)
            seq_type = [0]*(1 + len(seq[0])) + [1]*(len(seq[1]) + 1)
        else:
            seq_id = [self.map_dict['<cls>']] + [self.map_dict[x] for x in seq] + [self.map_dict['<eos>']]
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
                            constant_values=self.map_dict['<pad>'])
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
            
            if len(data['sequence_heavy']) + len(data['sequence_light']) + 2 > self.max_sl:
                data['sequence_light'] = data['sequence_light'][:self.max_sl - len(data['sequence_heavy']) - 2]
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
                 config = None,
                 map_dict = None,
                 encoder = None,
                 ):
        super().__init__()
        
        #self.mlm_probability = config['mlm_probability']
        
        #bert_config = BertConfig.from_json_file(config['bert_config'])
        #self.encoder = BertForMaskedModeling(config=bert_config)
        
        self.encoder = encoder
        self.max_sl = int(config['max_sl'])

        self.map_dict = map_dict

    def forward(self, input_ids, seq_masks, seq_types, targets):
                
        #mlm_output = self.encoder(input_ids = input_ids,
        #                          attention_mask = seq_masks,
        #                          return_dict = True,
        #                          labels = targets,
        #                          vocab_size = len(self.map_dict),
        #                          seq_types = seq_types
        #                        )
        
        mlm_output = self.encoder(input_ids, seq_types, repr_layers=[36])
        
        logits = mlm_output['logits']

        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(logits.view(-1, len(self.map_dict)), targets.view(-1))

        return masked_lm_loss
    
    def get_embedding(self, input_ids, seq_types):
        
        mlm_output = self.encoder(input_ids, seq_types, repr_layers=[36])
        return mlm_output['representations'][36]

def main(args, config):
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    alphabet='####LAGVSERTIDPKQNFYMHWCXBUZO.-##'

    #### Dataset ####
    print("Creating dataset")
    datasets = inference_dataset(config['test_file'], alphabet, config['max_sl'], device, map_dict)

    data_loader = DataLoader(
            datasets,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

    #### Model ####
    print("Creating model")
    
    model_data = torch.load(args.esm2_checkpoint)
    encoder, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
    #lora.mark_only_lora_as_trainable(encoder)
    ret = encoder.load_state_dict(model_state, strict=False)
    print('missed parameters',ret)
    model = Model(config=config, map_dict=map_dict, encoder=encoder)
    
    lora_weight = torch.load(config['ckpt_path'])['model']
    print(lora_weight.keys())
    ret = model.load_state_dict(lora_weight, strict=False)
    assert len(ret[1])==0
    for k in ret[0]:
        assert 'lora' not in k
    
    
    model = model.to(device)
    
    results=[]
    
    model.eval()
    with torch.no_grad():
        for i, (input_ids, seq_masks, seq_types) in tqdm(enumerate(data_loader)):
            
            input_ids=input_ids.to(device)
            seq_masks=seq_masks.to(device)
            seq_types=seq_types.to(device)
            last_hidden_state = model.get_embedding(input_ids, seq_types)
            last_hidden_state = last_hidden_state.cpu().data
            
            if config['if_pool']:
                assert config['how_pool'] in ['mean', 'max', 'cls']
                if config['how_pool'] == 'mean':
                    # mean pooling
                    seq_masks = seq_masks.cpu().data.unsqueeze(-1) # [B, L, 1]
                    seq_masks = seq_masks[:,1:,:]
                    last_hidden_state = last_hidden_state[:,1:,:]
                    
                    last_hidden_state = last_hidden_state * (seq_masks.repeat(1,1,2560))
                    last_hidden_state = last_hidden_state.sum(1, keepdim=True) / seq_masks.sum(1, keepdim=True) # [B, 1, 1024]
                elif config['how_pool'] == 'max':
                    # max pooling
                    seq_masks = seq_masks.cpu().data.unsqueeze(-1) # [B, L, 1]
                    seq_masks = seq_masks[:,1:,:]
                    last_hidden_state = last_hidden_state[:,1:,:]
                    
                    last_hidden_state = last_hidden_state + (1 - seq_masks.repeat(1,1,2560))*(-100)
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
    parser.add_argument('--esm2_checkpoint', default='code_dir/ab_downstream/ckpt/esm2_t36_3B_UR50D.pt') # esm2_t30_150M_UR50D.pt # esm2_t36_3B_UR50D.pt # esm2_t6_8M_UR50D
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=27, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    main(args, config)

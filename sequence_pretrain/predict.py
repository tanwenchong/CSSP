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
class predict_dataset(Dataset):
    def __init__(self, json_files, alphabet, max_sl, device, mlm_probability=0.15):
        
        # pdb_root: 'antibody_data/structure_data/'
        
        self.data_all, self.type_all = self._process_files(json_files)

        assert len(self.data_all)==len(self.type_all)
        
        self.parser = PDBParser(PERMISSIVE=1)
        self.alphabet = alphabet
        self.max_sl = max_sl
        self.mlm_probability = mlm_probability
        self.device=device
        self.map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}
        
        self.last_data=None
        
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

    def mask(self, input_ids, seq_cdr=None):
        targets=input_ids.clone()
        
        masked_indices = get_focused_span_mask(input_ids.numpy(), seq_cdr, self.mlm_probability) # targets [256, ]
        masked_indices = torch.tensor(masked_indices) # [280,], mask relative to origin target

        masked_indices[input_ids == self.map_dict['[CLS]']] = False
        masked_indices[input_ids == self.map_dict['[PAD]']] = False
        masked_indices[input_ids == self.map_dict['[SEP]']] = False

        targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.map_dict['[MASK]']

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(5, 24, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   

        return input_ids, targets.long()
    
    def preload(self, data, chain):
        if chain=='paired':
            seq_cdr = dict()
            for key in ['cdr1_heavy', 'cdr2_heavy', 'cdr3_heavy',
                        'cdr1_light', 'cdr2_light', 'cdr3_light']:
                if len(data[key]):
                    seq_cdr[key] = np.array([self.map_dict[x] for x in data[key]])
                else:
                    seq_cdr[key] = None

            seq_ids, seq_masks, seq_types = self.tokenize((data['sequence_heavy'], data['sequence_light']), self.max_sl, chain)
            
            ### cdr mask
            input_ids, targets = self.mask(seq_ids, seq_cdr)

            return input_ids, seq_masks, seq_types, targets
        
        else:
            seq_cdr = dict()
            for key in ['cdr1', 'cdr2', 'cdr3']:
                if len(data[key]):
                    seq_cdr[key] = np.array([self.map_dict[x] for x in data[key]])
                else:
                    seq_cdr[key] = None
            
            seq_ids, seq_masks, seq_types = self.tokenize(data['sequence'], max_sl = self.max_sl, chain_type = chain)
            
            ### cdr mask
            input_ids, targets = self.mask(seq_ids, seq_cdr)
            
            return input_ids, seq_masks, seq_types, targets


    def __getitem__(self, index):
        data, chain = self.data_all[index], self.type_all[index]

        if chain == 'paired':
            if len(data['sequence_heavy']) + len(data['sequence_light']) + 3 > self.max_sl: # save one place for cls
                data, chain = self.last_data
        else:
            if len(data['sequence']) + 2 > self.max_sl: # save one place for cls
                data, chain = self.last_data

        self.last_data = (data, chain)
        
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
        
        self.mlm_probability = config['mlm_probability']
        
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.encoder = BertForMaskedModeling(config=bert_config)
        self.max_sl = int(config['max_sl'])

        self.map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}


    def forward(self, input_ids, seq_masks, seq_types, targets):
        '''
        node_features: [B, L, 117]
        edge_features: [B, L, L, 272]
        label_masks: [B, L]
        '''
        
        
        mlm_output = self.encoder(input_ids = input_ids,
                                  attention_mask = seq_masks,
                                  return_dict = True,
                                  labels = targets,
                                  vocab_size = len(self.map_dict),
                                  seq_types = seq_types
                                )
        loss_mlm = mlm_output.loss
        logits_mlm = mlm_output.logits

        return loss_mlm, logits_mlm


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
    datasets = predict_dataset(config['test_file'], alphabet, config['max_sl'], device, config['mlm_probability'])

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
    model = Model(config=config)
    
    model = model.to(device)
    checkpoint = torch.load(config['ckpt_path'], map_location='cuda')
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    print('load checkpoint from %s'%config['ckpt_path'])
        
    model.eval()
    with torch.no_grad():
        for i, (input_ids, seq_masks, seq_types, targets) in tqdm(enumerate(data_loader)):
            
            input_ids=input_ids.to(device)
            seq_masks=seq_masks.to(device)
            seq_types=seq_types.to(device)
            targets=targets.to(device)
            loss, logits = model(input_ids, seq_masks, seq_types, targets)
            
            targets=targets.flatten()
            mask_position_index = (targets != -100).flatten()
            logits=logits.reshape(-1,logits.size(-1))
            predict_logits = logits[mask_position_index]
            true_label_id = targets[mask_position_index].cpu().numpy()
            predict_label_id = torch.argmax(predict_logits, dim=-1).detach().cpu().numpy()
            print('loss: {}, acc: {}'.format(loss.cpu().numpy(),(predict_label_id==true_label_id).sum()/len(true_label_id)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/predict.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint')
    parser.add_argument('--text_encoder', default='../pretrained/')
    parser.add_argument('--seed', default=27, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    main(args, config)

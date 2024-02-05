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

from dataset.utils import dict2feature, _get_features
from dataset.focused_mask import get_focused_span_mask
from models.modeling_bert import BertConfig, BertForMaskedModeling

from tqdm import tqdm

protein_letters_3to1['GAP']='X'

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}

def text_regular(seq):
    try:
        ret=re.sub(r'[\*\-]','',seq)
    except:
        print(seq)
        raise
    return ret

# single chain and paired chain
class inference_dataset(Dataset):
    def __init__(self, json_files, alphabet, pdb_root, topk, num_rbf_node, num_rbf_edge, max_sl, device):
        
        # pdb_root: 'antibody_data/structure_data/'
        
        self.data_all, self.type_all, self.idx_all = self._process_files(json_files)
        
        self.parser = PDBParser(PERMISSIVE=1)
        self.alphabet = alphabet
        self.pdb_root = pdb_root
        self.topk = topk
        self.num_rbf_node = num_rbf_node
        self.num_rbf_edge = num_rbf_edge
        self.max_sl = max_sl
        self.device=device
        self.map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}
        
    def _process_files(self, json_files):
        data_all=[]
        type_all=[]
        idx_all=[]
        for f in json_files:
            fname=os.path.basename(f)
            chain,idx=re.findall(r'([a-z]+)([0-9]*)\.json',fname)[0]
            data_=json.load(open(f,'r'))
            data_all+=data_
            type_all+=[chain]*len(data_)
            idx_all+=[str(idx)]*len(data_)
        return data_all, type_all, idx_all

    def __len__(self):
        return len(self.data_all)
    
    def _pdb2dict(self, pdb_path, chain = 'A', paired = False):
        if paired:
            residues = list(self.parser.get_structure('tp', pdb_path)[0].get_residues())
        else:
            residues = self.parser.get_structure('tp', pdb_path)[0][chain].get_list()

        seq_pdb=''
        res_dict=dict()
        res_dict['CA']=list()
        res_dict['C']=list()
        res_dict['N']=list()
        res_dict['O']=list()

        for res in residues:
            seq_pdb+=protein_letters_3to1[res.get_resname()]
            for atom in res:
                if atom.id in ['CA','C','N','O']:
                    res_dict[atom.id].append(atom.coord)

        res_dict['CA']=np.array(res_dict['CA'])
        res_dict['C']=np.array(res_dict['C'])
        res_dict['N']=np.array(res_dict['N'])
        res_dict['O']=np.array(res_dict['O'])
        res_dict['seq']=seq_pdb
        res_dict['file_name']=os.path.basename(pdb_path)
        
        return res_dict

    def tokenize(self, seq, max_sl, padding=True):
        if type(seq) == tuple:
            seq_id = [self.map_dict[x] for x in seq[0]] + [self.map_dict[x] for x in seq[1]]
            seq_mask = [1]*(1 + len(seq[0]) + len(seq[1]))
            seq_num = [0]*(1 + len(seq[0])) + [1]*len(seq[1])
        else:
            seq_id = [self.map_dict[x] for x in seq]
            seq_mask = [1]*(len(seq) + 1)
            seq_num = [0]*(len(seq) + 1)
        
        seq_id = np.array(seq_id)
        seq_mask = np.array(seq_mask)
        seq_num = np.array(seq_num)
        if padding:
            seq_id = np.pad(seq_id.astype(np.int32), (0, max_sl - len(seq_id)),
                            constant_values=self.map_dict['[PAD]']) # save one place for cls
            seq_mask = np.pad(seq_mask.astype(np.int32), (0, max_sl - len(seq_mask)), 
                              constant_values=0)
            seq_num = np.pad(seq_num.astype(np.int32), (0, max_sl - len(seq_num)), 
                              constant_values=0)
        
        # seq_id are origin target, seq_mask and seq_num contains cls
        return torch.tensor(seq_id), torch.tensor(seq_mask), torch.tensor(seq_num)
    
    def preload(self, data, pdb_root, topk, num_rbf_node, num_rbf_edge, chain, idx):
        
        pdb_path=os.path.join(pdb_root,data['pdb_file'])
        #pdb_path=os.path.join(pdb_root,chain,'split_predict',idx,data['pdb_file'])
        #print(chain)
        if chain=='paired':
            # paired seq
            seqsp, coords, maskp = dict2feature([self._pdb2dict(pdb_path, paired = True)], self.alphabet, self.max_sl)
            _, _, node_featuresp, edge_features = _get_features(seqsp, coords, maskp, topk, num_rbf_node, num_rbf_edge) # edge_featuresp as edge_features

            # heavy chain
            seqsh, coords, maskh = dict2feature([self._pdb2dict(pdb_path, chain = 'H')], self.alphabet, self.max_sl)
            _, _, node_featuresh, edge_featuresh = _get_features(seqsh, coords, maskh, topk, num_rbf_node, num_rbf_edge)

            # light chain
            seqsl, coords, maskl = dict2feature([self._pdb2dict(pdb_path, chain = 'L')], self.alphabet, self.max_sl)
            _, _, node_featuresl, edge_featuresl = _get_features(seqsl, coords, maskl, topk, num_rbf_node, num_rbf_edge)
            #print(len(data['sequence_heavy']),len(data['sequence_light']))
            label_ids, label_masks, seq_nums = self.tokenize((data['sequence_heavy'], data['sequence_light']), max_sl = self.max_sl)
            
            node_featuresh = node_featuresh[maskh.squeeze()==1]
            node_featuresl = node_featuresl[maskl.squeeze()==1]
            node_features = torch.cat([node_featuresh, 
                                       node_featuresl, 
                                       torch.zeros((self.max_sl - node_featuresh.size(0) - node_featuresl.size(0), node_featuresh.size(1))).to(node_featuresh.dtype)], axis=0) # [256, 117]
            
            return node_features, edge_features, label_masks, seq_nums
        
        else:
            seqs, coords, mask = dict2feature([self._pdb2dict(pdb_path, chain = 'A')], self.alphabet, self.max_sl)
            _, _, node_features, edge_features = _get_features(seqs, coords, mask, topk, num_rbf_node, num_rbf_edge)
            
            
            label_ids, label_masks, seq_nums = self.tokenize(data['sequence'], max_sl = self.max_sl)

            return node_features, edge_features, label_masks, seq_nums


    def __getitem__(self, index):
        data, chain, idx = self.data_all[index], self.type_all[index], self.idx_all[index]
        
        for key in data.keys():
        #for key in ['sequence','cdr']:
            if type(data[key])==str:
                data[key]=text_regular(data[key])

        if chain == 'paired':
            if len(data['sequence_heavy']) + len(data['sequence_light']) >= self.max_sl: # save one place for cls
                data['sequence_light'] = data['sequence_light'][self.max_sl - len(data['sequence_heavy']) - 1]
        else:
            if len(data['sequence']) >= self.max_sl: # save one place for cls
                data['sequence'] = data['sequence'][:self.max_sl - 1]

        ret = self.preload(data, self.pdb_root, self.topk, self.num_rbf_node, self.num_rbf_edge, chain, idx)

        return ret

class Model(nn.Module):
    def __init__(self,
                 config = None
                 ):
        super().__init__()
        
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.encoder = BertForMaskedModeling(config=bert_config)

        #self.node_feature_normalization = nn.BatchNorm1d(21 + config['num_rbf_node'] * 6)
        self.node_feature_transform = nn.Linear(21 + config['num_rbf_node'] * 6, bert_config.hidden_size)
        
        #self.edge_feature_normalization = nn.BatchNorm2d(16 + config['num_rbf_edge'] * 16)
        self.edge_feature_transform = nn.Linear(16 + config['num_rbf_edge'] * 16, bert_config.num_attention_heads)

        self.cls_node_feature = nn.Parameter(torch.randn((1, 1, bert_config.hidden_size)))
        self.topk = float(config['topk'])
        self.max_sl = int(config['max_sl'])
        self.edge_feature_dim = bert_config.num_attention_heads

        self.map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}


    def forward(self, nfeature, efeature, label_masks, seq_nums, bias_layer_num):
        '''
        node_features: [B, L, 117]
        edge_features: [B, L, L, 272]
        label_masks: [B, L]
        '''
        B = nfeature.size(0)
        #nfeature = self.node_feature_normalization(nfeature.permute(0,2,1)).permute(0,2,1)
        nfeature = self.node_feature_transform(nfeature)
        nfeature = torch.cat([self.cls_node_feature.repeat(B, 1, 1), nfeature], dim = 1)[:,:self.max_sl,:]
        #print(nfeature.shape)
        #print(nfeature[:,0,:]==self.cls_node_feature)

        #efeature = self.edge_feature_normalization(efeature.permute(0,3,1,2)).permute(0,2,3,1)
        efeature = self.edge_feature_transform(efeature).permute(0,3,1,2)
        # [B, 12, L, L]
        mask_2d = torch.matmul(label_masks.unsqueeze(2).float(), label_masks.unsqueeze(1).float()).unsqueeze(1) # [B, 1, L, L]
        efeature_cls = torch.sum(efeature * mask_2d, dim = 2, keepdim = True) # [B, 12, 1, L]
        mask_1d = label_masks.unsqueeze(1).unsqueeze(2).repeat(1, self.edge_feature_dim, 1, 1)
        efeature_cls_mean = torch.sum(efeature_cls * mask_1d.float(), dim = 3, keepdim = True) / (self.topk * torch.sum(mask_1d, dim = 3, keepdim=True)) # [B, 12, 1, 1]
        efeature = torch.cat([efeature_cls, efeature], dim = 2) # [B, 12, 1 + L, L]
        add_ = torch.cat([efeature_cls_mean, efeature_cls.permute(0,1,3,2)], dim = 2)
        efeature = torch.cat([add_, efeature], axis = 3) # [B, 12, 1 + L, 1 + L]
        efeature = efeature[:, :, :self.max_sl, :self.max_sl]

        #print(efeature.shape)
        
        mlm_output = self.encoder(nfeature,
                                  efeature,
                                  attention_mask = label_masks,
                                  return_dict = True,
                                  labels = None,
                                  vocab_size = len(self.map_dict),
                                  seq_nums = seq_nums,
                                  bias_layer_num = bias_layer_num,
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
    print(config['test_file'])
    datasets = inference_dataset(config['test_file'], alphabet, config['pdb_root'], config['topk'], config['num_rbf_node'], config['num_rbf_edge'], config['max_sl'], device)

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
    
    results=[]
    
    model.eval()
    with torch.no_grad():
        for i, (node_features, edge_features, label_masks, seq_nums) in tqdm(enumerate(data_loader)):
            #print(0)
            node_features=node_features.to(device)
            edge_features=edge_features.to(device)
            label_masks=label_masks.to(device)
            seq_nums=seq_nums.to(device)
            #print(0.5)
            last_hidden_state = model(node_features, edge_features, label_masks, seq_nums, config['bias_layer_num'])
            #print(last_hidden_state.shape)
            last_hidden_state = last_hidden_state.cpu().data
            #label_masks = label_masks.cpu().data.unsqueeze(-1) # [B, L, 1]
            
            
            if config['if_pool']:
                assert config['how_pool'] in ['mean','max','cls']
                if config['how_pool'] == 'mean':
                    # mean pooling

                    label_masks = label_masks.cpu().data.unsqueeze(-1) # [B, L, 1]
                    last_hidden_state = last_hidden_state[:,1:,:]
                    label_masks = label_masks[:,1:,:]
                    
                    last_hidden_state = last_hidden_state * (label_masks.repeat(1,1,768))
                    last_hidden_state = last_hidden_state.sum(1, keepdim=True) / label_masks.sum(1, keepdim=True) # [B, 1, 768]
                elif config['how_pool'] == 'max':
                    # max pooling

                    label_masks = label_masks.cpu().data.unsqueeze(-1) # [B, L, 1]
                    last_hidden_state = last_hidden_state[:,1:,:]
                    label_masks = label_masks[:,1:,:]
                    
                    last_hidden_state = last_hidden_state + (1 - label_masks.repeat(1,1,768))*(-100)
                    last_hidden_state = last_hidden_state.max(1, keepdim=True).values # [B, 1, 768]
                elif config['how_pool'] == 'cls':
                    # cls pooling
                    last_hidden_state = last_hidden_state[:,0,:]
            else:
                # no pooling
                last_hidden_state = last_hidden_state[:,:200,:]
                
            
            results.append(last_hidden_state.squeeze().numpy())
    results=np.concatenate(results,axis=0)
    #results=results.cpu().data.numpy()
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

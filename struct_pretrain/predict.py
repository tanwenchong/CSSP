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

# single chain and paired chain
class predict_dataset(Dataset):
    def __init__(self, json_files, alphabet, pdb_root, topk, num_rbf_node, num_rbf_edge, max_sl, device, mlm_probability=0.15):
        
        # pdb_root: 'antibody_data/structure_data/'
        
        self.data_all, self.type_all, self.idx_all = self._process_files(json_files)

        assert len(self.data_all)==len(self.type_all)
        assert len(self.data_all)==len(self.idx_all)

        
        self.parser = PDBParser(PERMISSIVE=1)
        self.alphabet = alphabet
        self.pdb_root = pdb_root
        self.topk = topk
        self.num_rbf_node = num_rbf_node
        self.num_rbf_edge = num_rbf_edge
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
        idx_all=[]
        for f in json_files:
            fname=os.path.basename(f)
            chain,idx=re.findall(r'([a-z]+)([0-9]+)\.json',fname)[0]
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

        #assert res_dict['CA'].shape==res_dict['C'].shape
        #assert res_dict['CA'].shape==res_dict['N'].shape
        #assert res_dict['CA'].shape==res_dict['O'].shape
        #assert res_dict['CA'].shape[0]==len(res_dict['seq'])
        
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

    def mask(self, nfeature, efeature, targets=None, seq_cdr=None):
        
        masked_indices = get_focused_span_mask(targets.numpy(), seq_cdr, self.mlm_probability) # targets [256, ]
        masked_indices = torch.tensor(masked_indices) # [280,], mask relative to origin target
        masked_indices[targets == self.map_dict['[PAD]']] = False
        #print(masked_indices.sum())

        # masked_indices are still relative to origin target
        save_indices = (~masked_indices).float() # [B, L]
        nfeature = nfeature * save_indices.unsqueeze(-1) # [L, D]
        save_indices = torch.matmul(save_indices.unsqueeze(-1),save_indices.unsqueeze(0)) # [L, L]
        efeature = efeature * save_indices.unsqueeze(-1) # [L, L, D] * [L, L, 1]


        # add place for cls, keep the length max_sl
        masked_indices = torch.cat([torch.tensor([False]), masked_indices], dim = 0)[:self.max_sl]
        targets = torch.cat([torch.tensor([-100]), targets], dim = 0)[:self.max_sl]
        
        targets[~masked_indices] = -100 # We only compute loss on masked tokens

        # nfeature and efeature are relative to origin target, and the cls will be concatenated in the training process
        return nfeature, efeature, targets.long()
    
    def preload(self, data, pdb_root, topk, num_rbf_node, num_rbf_edge, chain, idx):
        #seq_cdr = dict()
        #for key in ['cdr1', 'cdr2', 'cdr3']:
        #    seq_cdr[key] = data[key]
        
        pdb_path=os.path.join(pdb_root,chain,'split_predict',idx,data['pdb_file'])
        #print(chain)
        if chain=='paired':
            seq_cdr = dict()
            #for key in ['cdr1_heavy', 'cdr2_heavy', 'cdr3_heavy',
            #            'cdr1_light', 'cdr2_light', 'cdr3_light']:
            for key in ['cdr3_heavy',
                        'cdr3_light']:
                if len(data[key]):
                    tp_, _, _ = self.tokenize(data[key], max_sl = len(data[key]), padding = False)
                    seq_cdr[key] = tp_.numpy()
                else:
                    seq_cdr[key] = None
            
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

            ### cdr mask
            nfeature, efeature, labels = self.mask(node_features, edge_features, targets = label_ids, seq_cdr = seq_cdr)
            #print(label_ids)
            #print(labels)
            
            return nfeature, efeature, labels, label_masks, seq_nums
        
        else:
            seq_cdr = dict()
            for key in ['cdr3']:
                if len(data[key]):
                    tp_, _, _ = self.tokenize(data[key], max_sl = len(data[key]), padding = False)
                    seq_cdr[key] = tp_.numpy()
                else:
                    seq_cdr[key] = None
            
            seqs, coords, mask = dict2feature([self._pdb2dict(pdb_path, chain = 'A')], self.alphabet, self.max_sl)
            _, _, node_features, edge_features = _get_features(seqs, coords, mask, topk, num_rbf_node, num_rbf_edge)
            
            
            label_ids, label_masks, seq_nums = self.tokenize(data['sequence'], max_sl = self.max_sl)
            
            
            ### cdr mask
            nfeature, efeature, labels = self.mask(node_features, edge_features, targets = label_ids, seq_cdr = seq_cdr)
            #print(label_ids)
            #print(labels)
            
            #print(nfeature.shape)

            return nfeature, efeature, labels, label_masks, seq_nums


    def __getitem__(self, index):
        data, chain, idx = self.data_all[index], self.type_all[index], self.idx_all[index]
        
        #print('rank {}, idx {}, chain {}'.format(get_rank(),index,chain))

        if chain == 'paired':
            if len(data['sequence_heavy']) + len(data['sequence_light']) >= self.max_sl: # save one place for cls
                data, chain, idx = self.last_data
        else:
            if len(data['sequence']) >= self.max_sl: # save one place for cls
                data, chain, idx = self.last_data

        self.last_data = (data, chain, idx)
        
        #if chain=='paired':
        #    print(chain, len(data['sequence_heavy'])+len(data['sequence_light']))
        #else:
        #    print(chain, len(data['sequence']))

        ret = self.preload(data, self.pdb_root, self.topk, self.num_rbf_node, self.num_rbf_edge, chain, idx)

        return ret

class Model(nn.Module):
    def __init__(self,
                 config = None
                 ):
        super().__init__()
        
        self.mlm_probability = config['mlm_probability']
        
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


    def forward(self, nfeature, efeature, labels, label_masks, seq_nums, bias_layer_num):
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
                                  labels = labels,
                                  vocab_size = len(self.map_dict),
                                  seq_nums = seq_nums,
                                  bias_layer_num = bias_layer_num,
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
    datasets = predict_dataset(config['test_file'], alphabet, config['pdb_root'], config['topk'], config['num_rbf_node'], config['num_rbf_edge'], config['max_sl'], device, config['mlm_probability'])

    data_loader = DataLoader(
            datasets,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            shuffle=True,
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
        for i, (node_features, edge_features, labels, label_masks, seq_nums) in tqdm(enumerate(data_loader)):
            if i>=5000:
                break
            node_features=node_features.to(device)
            edge_features=edge_features.to(device)
            labels=labels.to(device)
            label_masks=label_masks.to(device)
            seq_nums=seq_nums.to(device)
            loss, logits = model(node_features, edge_features, labels, label_masks, seq_nums, config['bias_layer_num'])
            
            
            labels=labels.flatten()
            mask_position_index = (labels != -100).flatten()
            logits=logits.reshape(-1,logits.size(-1))
            predict_logits = logits[mask_position_index]
            true_label_id = labels[mask_position_index].cpu().numpy()
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

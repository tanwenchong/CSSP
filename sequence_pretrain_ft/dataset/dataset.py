import os
import json
import re

import random

import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import Dataset

from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1

from dataset.focused_mask import get_focused_span_mask

protein_letters_3to1['GAP']='X'

def text_regular(seq):
    return re.sub(r'[\*\-]','',seq)

# single chain and paired chain
class pretrain_dataset(Dataset):
    def __init__(self, json_files, alphabet, max_sl, device, map_dict, mlm_probability=0.15):
        
        # pdb_root: 'antibody_data/structure_data/'
        
        self.data_all, self.type_all = self._process_files(json_files)
        
        self.data_all = self.data_all
        self.type_all = self.type_all
        
        print('dataset length',len(self.data_all))

        assert len(self.data_all)==len(self.type_all)
        
        self.parser = PDBParser(PERMISSIVE=1)
        self.alphabet = alphabet
        self.max_sl = max_sl
        self.mlm_probability = mlm_probability
        self.device=device
        self.map_dict = map_dict
        # {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
        self.last_data=None
        
    def _process_files(self, json_files):
        data_all=[]
        type_all=[]
        for f in json_files:
            fname=os.path.basename(f)
            chain,idx=re.findall(r'([a-z]+)([0-9\-]*).*\.json',fname)[0]
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

    def mask(self, input_ids, seq_cdr=None):
        targets=input_ids.clone()
        
        masked_indices = get_focused_span_mask(input_ids.numpy(), seq_cdr, self.mlm_probability) # targets [256, ]
        masked_indices = torch.tensor(masked_indices) # [280,], mask relative to origin target

        masked_indices[input_ids == self.map_dict['<cls>']] = False
        masked_indices[input_ids == self.map_dict['<pad>']] = False
        masked_indices[input_ids == self.map_dict['<eos>']] = False

        targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.map_dict['<mask>']

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
            
            #print(input_ids)
            #print(targets)
            
            return input_ids, seq_masks, seq_types, targets


    def __getitem__(self, index):
        data, chain = self.data_all[index], self.type_all[index]
        
        for key in data.keys():
            data[key]=text_regular(data[key])


        if chain == 'paired':
            if len(data['sequence_heavy']) + len(data['sequence_light']) + 2 > self.max_sl:
                data, chain = self.last_data
            
            # check X proportion
            elif (data['sequence_heavy'].count('X') + data['sequence_light'].count('X')) / (len(data['sequence_heavy']) + len(data['sequence_light'])) > 0.3:
                data, chain = self.last_data
        
        else:
            if len(data['sequence']) + 2 > self.max_sl:
                data, chain = self.last_data

            # check X proportion
            elif data['sequence'].count('X') / len(data['sequence']) > 0.3:
                data, chain = self.last_data

        self.last_data = (data, chain)
        
        #if chain=='paired':
        #    print(chain, len(data['sequence_heavy'])+len(data['sequence_light']))
        #else:
        #    print(chain, len(data['sequence']))

        ret = self.preload(data, chain)

        return ret







class test_dataset(Dataset):
    def __init__(self, json_files, alphabet, max_sl, device, mlm_probability=0.15):
        
        self.data_all, self.type_all = self._process_files(json_files)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):

        return torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0])
    
    def _process_files(self, json_files):
        data_all=[]
        type_all=[]
        for f in json_files:
            fname=os.path.basename(f)
            chain,idx=re.findall(r'([a-z]+)([0-9\-]*).*\.json',fname)[0]
            assert chain in ['heavy','light','paired']
            data_=json.load(open(f,'r'))
            data_all+=data_
            type_all+=[chain]*len(data_)
        return data_all, type_all





## single chain
'''
class pretrain_dataset(Dataset):
    def __init__(self, json_files, alphabet, pdb_root, topk, num_rbf, max_sl, device, mlm_probability=0.15):
        
        # pdb_root: 'antibody_data/structure_data/'
        
        #self.data_all = []
        #for f in json_files:
        #    self.data_all += json.load(open(f,'r'))
        self.data_all, self.type_all, self.idx_all = self._process_files(json_files)

        assert len(self.data_all)==len(self.type_all)
        assert len(self.data_all)==len(self.idx_all)

        
        self.parser = PDBParser(PERMISSIVE=1)
        self.alphabet = alphabet
        self.pdb_root = pdb_root
        self.topk = topk
        self.num_rbf = num_rbf
        self.max_sl = max_sl
        self.mlm_probability = mlm_probability
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
            chain,idx=re.findall(r'([a-z]+)([0-9]+)\.json',fname)[0]
            data_=json.load(open(f,'r'))
            data_all+=data_
            type_all+=[chain]*len(data_)
            idx_all+=[str(idx)]*len(data_)
        return data_all, type_all, idx_all

    def __len__(self):
        return len(self.data_all)
    
    def _pdb2dict(self, pdb_path):
        residues = self.parser.get_structure('tp', pdb_path)[0]['A'].get_list()

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

        assert res_dict['CA'].shape==res_dict['C'].shape
        assert res_dict['CA'].shape==res_dict['N'].shape
        assert res_dict['CA'].shape==res_dict['O'].shape
        assert res_dict['CA'].shape[0]==len(res_dict['seq'])
        
        return res_dict

    def tokenize(self, seq, max_sl):
        seq_id = [self.map_dict[x] for x in seq]
        seq_id = np.array(seq_id)
        seq_mask = np.pad(np.ones(seq_id.shape, dtype=np.int32), (0, max_sl - len(seq)),
                            constant_values=0)
        seq_id = np.pad(seq_id.astype(np.int32), (0, max_sl - len(seq)),
                          constant_values=self.map_dict['[PAD]'])
        return torch.tensor(seq_id), torch.tensor(seq_mask)

    def mask(self, nfeature, efeature, targets=None, probability_matrix=None):
        #return nfeature, efeature, targets.long()
        
        ### slow ?
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[targets == self.map_dict['[PAD]']] = False
        
        targets[~masked_indices] = -100 # We only compute loss on masked tokens

        save_indices = (~masked_indices).float() # [B, L]
        
        nfeature = nfeature * save_indices.unsqueeze(-1) # [L, D]
        
        save_indices = torch.matmul(save_indices.unsqueeze(-1),save_indices.unsqueeze(0)) # [L, L]
        
        efeature = efeature * save_indices.unsqueeze(-1) # [L, L, D] * [L, L, 1]
        #efeature = efeature * save_indices.unsqueeze(1).unsqueeze(-1) # [L, L, D] * [1, L, 1]
        #efeature = efeature.permute(1,0,2) * save_indices.unsqueeze(-1).unsqueeze(-1) # [L, L, D] * [1, L, 1]
        #efeature = efeature.permute(1,0,2)
        
        #print(nfeature.shape)
        #print(efeature.shape)

        return nfeature, efeature, targets.long()
    
    def preload(self, data, pdb_root, topk, num_rbf, chain, idx):
        seq_cdr = dict()
        for key in ['cdr1', 'cdr2', 'cdr3']:
            seq_cdr[key] = data[key]
        
        pdb_path=os.path.join(pdb_root,chain,'split_predict',idx,data['pdb_file'])
        seqs, coords, mask = dict2feature([self._pdb2dict(pdb_path)], self.alphabet, self.max_sl)
        coords, seqs, node_features, edge_features = _get_features(seqs, coords, mask, topk, num_rbf)
        
        label_ids, label_masks = self.tokenize(data['sequence'], max_sl = self.max_sl)
        probability_matrix = torch.full(label_ids.shape, self.mlm_probability)
        
        ### cdr mask
        nfeature, efeature, labels = self.mask(node_features, edge_features, targets=label_ids,
                                              probability_matrix = probability_matrix)

        return nfeature, efeature, labels, label_masks


    def __getitem__(self, index):
        data, chain, idx = self.data_all[index], self.type_all[index], self.idx_all[index]
        
        ret = self.preload(data, self.pdb_root, self.topk, self.num_rbf, chain, idx)

        return ret
'''
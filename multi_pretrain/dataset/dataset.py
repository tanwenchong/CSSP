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

from dataset.utils import dict2feature, _get_features
from dataset.focused_mask import get_focused_span_mask

protein_letters_3to1['GAP']='X'

def text_regular(seq):
    return re.sub(r'[\*\-]','',seq)

# single chain and paired chain
class pretrain_dataset(Dataset):
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
        self.map_dict_r = {v:k for k,v in self.map_dict.items()}
        
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
        
        return res_dict

    def tokenize_stru(self, seq, max_sl, padding=True, add_sep_cls = False):
        # ATTENTION: THIS TOKENIZE FUNCTION IS NOT THE SAME AS THAT FOR STRUCTURE PRETRAIN, SEQ_ID IS CORRESPONDING WITH TOKENIZE_SEQ TO GENERATE FOCUSED MASK
        if type(seq) == tuple:
            # This changing doesn't affect seq_mask and seq_num which are used to extract structure feature
            if add_sep_cls:
                seq_id = [self.map_dict['[CLS]']] + [self.map_dict[x] for x in seq[0]] + [self.map_dict['[SEP]']] + [self.map_dict[x] for x in seq[1]] + [self.map_dict['[SEP]']]
            else:
                seq_id = [self.map_dict[x] for x in seq[0]] + [self.map_dict[x] for x in seq[1]]

            # keep the same as those in structure pretrain
            seq_mask = [1]*(1 + len(seq[0]) + len(seq[1]))
            seq_num = [0]*(1 + len(seq[0])) + [1]*len(seq[1])
        else:
            # This changing doesn't affect seq_mask and seq_num which are used to extract structure feature
            if add_sep_cls:
                seq_id = [self.map_dict['[CLS]']] + [self.map_dict[x] for x in seq] + [self.map_dict['[SEP]']]
            else:
                seq_id = [self.map_dict[x] for x in seq]

            # keep the same as those in structure pretrain
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

    def tokenize_seq(self, seq, max_sl, chain_type, padding=True):
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
    
    def id2token(self, ids):
        return ''.join([self.map_dict_r[i] for i in ids])

    def mask(self, input_ids, input_ids_stru, seq_cdr=None):
        targets=input_ids.clone()
        
        masked_indices = get_focused_span_mask(input_ids_stru.numpy(), seq_cdr, self.mlm_probability) # targets [256, ]
        masked_indices = torch.tensor(masked_indices) # [280,], mask relative to origin target

        masked_indices[input_ids_stru == self.map_dict['[CLS]']] = False
        masked_indices[input_ids_stru == self.map_dict['[PAD]']] = False
        masked_indices[input_ids_stru == self.map_dict['[SEP]']] = False

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
    
    def preload(self, data, pdb_root, topk, num_rbf_node, num_rbf_edge, chain, idx):
        
        pdb_path=os.path.join(pdb_root,chain,'split_predict',idx,data['pdb_file'])
        if chain=='paired':

            seq_cdr = dict()
            for key in ['cdr1_heavy', 'cdr2_heavy', 'cdr3_heavy',
                        'cdr1_light', 'cdr2_light', 'cdr3_light']:
                if len(data[key]):
                    tp_, _, _ = self.tokenize_stru(data[key], max_sl = len(data[key]), padding = False)
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

            # label_ids here is only used to generate focused mask
            label_ids, label_masks, seq_nums = self.tokenize_stru((data['sequence_heavy'], data['sequence_light']), max_sl = self.max_sl, add_sep_cls=True)
            
            #print(seq_cdr)
            #print(label_ids)


            seqsh = seqsh.squeeze().numpy()
            maskh = maskh.squeeze().numpy().astype(bool)
            seqsh = self.id2token(seqsh[maskh])
            seqsl = seqsl.squeeze().numpy()
            maskl = maskl.squeeze().numpy().astype(bool)
            seqsl = self.id2token(seqsl[maskl])
            seq_ids, seq_masks, seq_types = self.tokenize_seq((seqsh, seqsl), max_sl = self.max_sl, chain_type = chain)

            #input_ids, targets = self.mask(seq_ids.clone(), label_ids, seq_cdr)
            
            node_featuresh = node_featuresh[maskh.squeeze()==1]
            node_featuresl = node_featuresl[maskl.squeeze()==1]
            node_features = torch.cat([node_featuresh, 
                                       node_featuresl, 
                                       torch.zeros((self.max_sl - node_featuresh.size(0) - node_featuresl.size(0), node_featuresh.size(1))).to(node_featuresh.dtype)], axis=0) # [256, 117]
            
            #print(seq_ids)
            #print(input_ids)
            #print(targets)
            
            #np.save('nfeaturep.npy',node_features.numpy())
            #np.save('efeaturep.npy',edge_features.numpy())

            ### cdr mask
            #nfeature, efeature, labels = self.mask(node_features, edge_features, targets = label_ids, seq_cdr = seq_cdr)
            
            #return node_features, edge_features, label_masks, seq_nums, seq_ids, input_ids, seq_masks, seq_types, targets
            return node_features, edge_features, label_masks, seq_nums, seq_ids, seq_masks, seq_types
        
        
        else:
            seq_cdr = dict()
            for key in ['cdr1', 'cdr2', 'cdr3']:
                if len(data[key]):
                    tp_, _, _ = self.tokenize_stru(data[key], max_sl = len(data[key]), padding = False)
                    seq_cdr[key] = tp_.numpy()
                else:
                    seq_cdr[key] = None

            seqs, coords, mask = dict2feature([self._pdb2dict(pdb_path, chain = 'A')], self.alphabet, self.max_sl)
            _, _, node_features, edge_features = _get_features(seqs, coords, mask, topk, num_rbf_node, num_rbf_edge)
            
            #np.save('nfeature.npy',node_features.numpy())
            #np.save('efeature.npy',edge_features.numpy())
            
            # label_ids here is only used to generate focused mask
            label_ids, label_masks, seq_nums = self.tokenize_stru(data['sequence'], max_sl = self.max_sl, add_sep_cls=True)
            
            #print(seq_cdr)
            #print(label_ids)
            

            seqs = seqs.squeeze().numpy()
            mask = mask.squeeze().numpy().astype(bool)
            seqs = self.id2token(seqs[mask])
            seq_ids, seq_masks, seq_types = self.tokenize_seq(seqs, max_sl = self.max_sl, chain_type = chain)
            
            ### cdr mask
            #nfeature, efeature, labels = self.mask(node_features, edge_features, targets = label_ids, seq_cdr = seq_cdr)
            #input_ids, targets = self.mask(seq_ids.clone(), label_ids, seq_cdr)
            
            #print(seq_ids)
            #print(input_ids)
            #print(targets)
            
            #print(nfeature.shape)

            #return node_features, edge_features, label_masks, seq_nums, seq_ids, input_ids, seq_masks, seq_types, targets
            return node_features, edge_features, label_masks, seq_nums, seq_ids, seq_masks, seq_types

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

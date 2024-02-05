'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
#from models.xbert import BertConfig, BertForMaskedLM
from models.modeling_bert import BertConfig, BertForMaskedModeling

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
import time

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

        return loss_mlm

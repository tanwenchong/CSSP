'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial

from models.modeling_bert_seq import BertConfig as BertConfigSeq
from models.modeling_bert_stru import BertConfig as BertConfigStru

from models.modeling_bert_seq import BertForMaskedModeling as BertForMaskedModelingSeq
from models.modeling_bert_stru import BertForMaskedModeling as BertForMaskedModelingStru

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random

class SeqEncoder(nn.Module):
    def __init__(self,
                 config = None
                 ):
        super().__init__()
        
        #self.mlm_probability = config['mlm_probability_seq']
        
        bert_config = BertConfigSeq.from_json_file(config['bert_config_seq'])
        self.encoder = BertForMaskedModelingSeq(config=bert_config)
        self.max_sl = int(config['max_sl'])
        
        self.hidden_size = bert_config.hidden_size

        self.map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}


    def forward(self, input_ids, seq_masks, seq_types):
        hidden_state = self.encoder(input_ids = input_ids,
                                  attention_mask = seq_masks,
                                  return_dict = True,
                                  labels = None,
                                  vocab_size = len(self.map_dict),
                                  seq_types = seq_types
                                )

        return hidden_state #mlm_output.hidden_states

class StruEncoder(nn.Module):
    def __init__(self,
                 config = None
                 ):
        super().__init__()
        
        bert_config = BertConfigStru.from_json_file(config['bert_config_stru'])
        
        self.encoder = BertForMaskedModelingStru(config=bert_config)
        
        self.hidden_size = bert_config.hidden_size

        self.node_feature_normalization = nn.BatchNorm1d(21 + config['num_rbf_node'] * 6)
        self.node_feature_transform = nn.Linear(21 + config['num_rbf_node'] * 6, bert_config.hidden_size)
        
        self.edge_feature_normalization = nn.BatchNorm2d(16 + config['num_rbf_edge'] * 16)
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
        nfeature = self.node_feature_normalization(nfeature.permute(0,2,1)).permute(0,2,1)
        nfeature = self.node_feature_transform(nfeature)
        nfeature = torch.cat([self.cls_node_feature.repeat(B, 1, 1), nfeature], dim = 1)[:,:self.max_sl,:]
        #print(nfeature.shape)
        #print(nfeature[:,0,:]==self.cls_node_feature)

        efeature = self.edge_feature_normalization(efeature.permute(0,3,1,2)).permute(0,2,3,1)
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
        
        hidden_state = self.encoder(nfeature,
                                  efeature,
                                  attention_mask = label_masks,
                                  return_dict = True,
                                  labels = None,
                                  vocab_size = len(self.map_dict),
                                  seq_nums = seq_nums,
                                  bias_layer_num = bias_layer_num,
                                )

        return hidden_state #mlm_output.hidden_states


class CSSP(nn.Module):
    def __init__(self,
                 config = None,
                 ):
        super().__init__()

        self.seq_encoder=SeqEncoder(config)
        self.stru_encoder=StruEncoder(config)
        
        #self.seq_projection = nn.Parameter(torch.randn(self.seq_encoder.hidden_size, config['embed_dim']))
        #self.stru_projection = nn.Parameter(torch.randn(self.stru_encoder.hidden_size, config['embed_dim']))
        
        
        self.seq_projection = nn.Sequential(
            nn.Linear(self.seq_encoder.hidden_size, self.seq_encoder.hidden_size),
            nn.ReLU(),
            nn.Linear(self.seq_encoder.hidden_size, config['embed_dim'])
        )
        self.stru_projection = nn.Sequential(
            nn.Linear(self.stru_encoder.hidden_size, self.stru_encoder.hidden_size),
            nn.ReLU(),
            nn.Linear(self.stru_encoder.hidden_size, config['embed_dim'])
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, nfeature, efeature, stru_label_masks, stru_seq_nums, 
                seq_input_ids_contrast, 
                seq_masks, seq_types, 
                stru_bias_layer_num):

        # contrastive loss
        seq_f=self.seq_encoder(seq_input_ids_contrast, seq_masks, seq_types)
        stru_f=self.stru_encoder(nfeature, efeature, stru_label_masks, stru_seq_nums, stru_bias_layer_num)
        
        #seq_f = seq_f[:,0,:] @ self.seq_projection
        #stru_f = stru_f[:,0,:] @ self.stru_projection
        
        seq_f = self.seq_projection(seq_f[:,0,:])
        stru_f = self.stru_projection(stru_f[:,0,:])
        
        # normalized features
        seq_f = seq_f / seq_f.norm(dim=1, keepdim=True)
        stru_f = stru_f / stru_f.norm(dim=1, keepdim=True)
        
        logits = (seq_f @ stru_f.t()) * self.logit_scale.exp()
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        loss_ita = (self.cross_entropy_loss(logits, labels) + self.cross_entropy_loss(logits.t(), labels)) / 2
        
        
        # mlm loss
        #_, loss_mlm = self.seq_encoder(seq_input_ids_mlm, seq_masks, seq_types, seq_targets)

        return loss_ita

    def get_seq_embedding(self, input_ids, seq_masks, seq_types):
        return self.seq_encoder(input_ids, seq_masks, seq_types)
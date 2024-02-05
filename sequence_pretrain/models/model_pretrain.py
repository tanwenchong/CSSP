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

        return loss_mlm

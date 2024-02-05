'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
#from models.xbert import BertConfig, BertForMaskedLM
#from models.modeling_bert import BertConfig, BertForMaskedModeling

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

import numpy as np
import random
import time

from .esm2 import ESM2

class Model(nn.Module):
    def __init__(self,
                 config = None,
                 map_dict = None,
                 encoder = None,
                 ):
        super().__init__()
        
        self.mlm_probability = config['mlm_probability']
        
        #bert_config = BertConfig.from_json_file(config['bert_config'])
        #self.encoder = BertForMaskedModeling(config=bert_config)
        
        self.encoder = encoder
        self.max_sl = int(config['max_sl'])
        self.loss_fct = CrossEntropyLoss()  # -100 index = padding token

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

        
        masked_lm_loss = self.loss_fct(logits.view(-1, len(self.map_dict)), targets.view(-1))
        

        return masked_lm_loss

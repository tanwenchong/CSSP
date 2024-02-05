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
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_pretrain import CSSP
from models.esm2 import ESM2

import utils
from dataset import create_dataset, create_sampler, create_loader
from dataset.data import Alphabet
from scheduler import create_scheduler
from optim import create_optimizer

import re
import time

from tqdm import tqdm

import loralib as lora

vocabs = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

seq_map_dict = dict()

for i in range(len(vocabs)):
    seq_map_dict[vocabs[i]] = i



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


def part_state_dict(my_state_dict, grad_keys):
    return {k: my_state_dict[k] for k in my_state_dict if k in grad_keys}


def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 1000
    warmup_iterations = warmup_steps*step_size
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
        

    for i, (nfeature, efeature, stru_label_masks, stru_seq_nums, seq_input_ids_contrast, seq_masks, seq_types) in tqdm(enumerate(metric_logger.log_every(data_loader, print_freq, header))):
        
        optimizer.zero_grad()
        loss = model(nfeature, efeature, stru_label_masks, stru_seq_nums, 
                        seq_input_ids_contrast, 
                         seq_masks, seq_types, 
                        config['bias_layer_num'])

        
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch==0 and i%step_size==0 and i<=warmup_iterations:  #### ?
            scheduler.step(i//step_size)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
    
    
def main(args, config):
    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    seq_alphabet='####LAGVSERTIDPKQNFYMHWCXBUZO.-##'
    stru_alphabet='#####LAGVESIKRDTPNQFYMHCWX'

    #### Dataset ####
    print("Creating dataset")
    datasets = [create_dataset(config, stru_alphabet, device, seq_map_dict)]
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']], num_workers=[10], is_trains=[True], collate_fns=[None])[0]





    model_data = torch.load(args.esm2_checkpoint)
    seq_encoder, _, model_state = _load_model_and_alphabet_core_v2(model_data)
    lora.mark_only_lora_as_trainable(seq_encoder)
    ret = seq_encoder.load_state_dict(model_state, strict=False)
    #print('missed parameters',ret)
    assert len(ret[1]) == 0
    for ret_ in ret[0]:
        assert ('lora' in ret_) or ('embed_types' in ret_) or ('embed_tokens' in ret_)
    
    for param in seq_encoder.named_parameters():
        if ('embed_types' in param[0]) or ('embed_tokens' in param[0]):
            param[1].requires_grad=True
    
    param_to_train=[]
    params_all=seq_encoder.named_parameters()
    for param in params_all:
        if param[1].requires_grad:
            param_to_train.append(param[0])
    print('parameters to train',param_to_train)
    
    
    #### Model ####
    print("Creating model")
    model = CSSP(config=config, seq_map_dict=seq_map_dict, seq_encoder=seq_encoder)
    stru_ckpt = torch.load(args.stru_checkpoint)['model']
    ret = model.stru_encoder.load_state_dict(stru_ckpt, strict=False)
    assert len(ret[0]) == 0
    for k in ret[1]:
        assert 'cls' in k

    model = model.to(device)
    
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1
        ret = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s'%args.checkpoint)
        #print('missed keys when loading lora', ret)
        print(ret)
        assert len(ret[1]) == 0
#        for ret_ in ret[0]:
#            assert ('lora' not in ret_) and ('embed_types' not in ret_) and ('embed_tokens' not in ret_)
        
        
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params: {:.3f} M'.format(n_parameters/(1024*1024)))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params: {:.3f} M'.format(n_parameters/(1024*1024)))
    
    grad_keys = []
    for param_tuple in model.named_parameters():
        name, param = param_tuple
        if param.requires_grad:
            grad_keys.append(name)


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu],
                                                          #find_unused_parameters=True
                                                         )
        model_without_ddp = model.module
        
    #print(part_state_dict(model_without_ddp.state_dict(), grad_keys).keys())
    
    
    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        
        train_stats = train(model, data_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)
        
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }
            save_obj = {
                'model': part_state_dict(model_without_ddp.state_dict(), grad_keys),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--esm2_checkpoint', default='code_dir/ab_downstream/ckpt/esm2_t36_3B_UR50D.pt')
    parser.add_argument('--stru_checkpoint', default='antibody_pretrain/structure_pretrain/struct_pretrain/output/Pretrain/checkpoint_24.pth')
    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='../pretrained/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=27, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default=None, help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args, config)
    
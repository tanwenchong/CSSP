import os
import re
import numpy as np

import matplotlib.pyplot as plt


if 1:
    f=open('antibody_pretrain/structure_pretrain/struct_code4/output_test1_4.txt','r')
    lines=f.readlines()
    f.close()

    smooth=True
    step=1

    loss=[]

    if not smooth:
        for line in lines:
            find=re.findall(r'lr: ([0-9\.]+)  loss',line)
            if len(find):
                loss.append(float(find[0]))
        plt.plot(loss)
        plt.savefig('lr.png')
    else:
        count=0
        tp_=[]
        for line in lines:
            find=re.findall(r'lr: ([0-9\.]+)  loss',line)
            if len(find):
                tp_.append(float(find[0]))
                if count%step==0:
                    loss.append(np.mean(tp_))
                    tp_=[]
                count+=1
        plt.plot(loss)
        plt.savefig('lr_step{}.png'.format(step))


def get_loss(path, smooth, step):
    f=open(path,'r')
    lines=f.readlines()
    f.close()
    loss=[]

    if not smooth:
        for line in lines:
            find=re.findall(r'[0-9]+it.+loss: ([0-9\.]+)  time:',line)
            if len(find):
                loss.append(float(find[0]))
    else:
        count=0
        tp_=[]
        for line in lines:
            find=re.findall(r'[0-9]+it.+loss: ([0-9\.]+)  time:',line)
            if len(find):
                tp_.append(float(find[0]))
                if count%step==0:
                    loss.append(np.mean(tp_))
                    tp_=[]
                count+=1
    return loss


if 0:
    smooth=True
    step=10
    
    loss=get_loss('antibody_pretrain/structure_pretrain/struct_code4/output_test1_4.txt',smooth,step)
    loss_nobn=get_loss('antibody_pretrain/structure_pretrain/struct_pretrain/output_test1_4_nobn.txt',smooth,step)
    loss_nosne=get_loss('antibody_pretrain/structure_pretrain/struct_code4_nosne/output_test1_4_nosne.txt',smooth,step)
    plt.figure()
    plt.plot(loss,label='all')
    plt.plot(loss_nobn,label='nobn')
    plt.plot(loss_nosne,label='nosne')
    #plt.xlim(0,50)
    plt.ylim(0.4,1.0)
    plt.legend()
    plt.savefig('loss_step{}.png'.format(step))
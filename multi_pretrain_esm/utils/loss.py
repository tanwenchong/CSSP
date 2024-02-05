import os
import re
import numpy as np

import matplotlib.pyplot as plt


if 0:
    f=open('antibody_pretrain/sequence_pretrain_ft/output_8_3B_ft.txt','r')
    lines=f.readlines()
    f.close()

    smooth=True
    step=10

    loss=[]

    if not smooth:
        for line in lines:
            find=re.findall(r'[0-9]+it.+loss: ([0-9\.]+)  time:',line)
            if len(find):
                loss.append(float(find[0]))
        plt.plot(loss)
        plt.savefig('loss.png')
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
        plt.plot(loss)
        plt.ylim(0.8,1.2)
        plt.savefig('loss_step{}.png'.format(step))

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

def get_loss_resume(path, smooth, step):
    lines=[]
    for pa_ in path:
        f=open(pa_,'r')
        add_=f.readlines()
        print(len(add_))
        lines+=add_
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

if 1:
    smooth=True
    step=1
    
    loss=get_loss_resume(['antibody_pretrain/multi_pretrain_esm/output_4.txt'],smooth,step)
    plt.figure()
    plt.plot(loss)
    plt.savefig('loss_step{}.png'.format(step))




if 0:
    smooth=True
    step=1
    
    loss=get_loss('antibody_pretrain/structure_pretrain/struct_code4/output_test1_4.txt',smooth,step)
    loss_nobn=get_loss('antibody_pretrain/structure_pretrain/struct_code4_nobn/output_test1_4_nobn.txt',smooth,step)
    loss_nosne=get_loss('antibody_pretrain/structure_pretrain/struct_code4_nosne/output_test1_4_nosne.txt',smooth,step)
    plt.figure()
    plt.plot(loss,label='all')
    plt.plot(loss_nobn,label='nobn')
    plt.plot(loss_nosne,label='nosne')
    plt.legend()
    plt.savefig('loss_step{}.png'.format(step))
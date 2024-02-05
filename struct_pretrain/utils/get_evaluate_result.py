import os
import re
import numpy as np


root='antibody_pretrain/structure_pretrain/struct_pretrain/output/evaluate_cdr3/'


for i in range(25):
    path=os.path.join(root,'output_cdr3_{}.txt'.format('0'*(2-len(str(i)))+str(i)))
    f=open(path,'r')
    lines=f.readlines()
    f.close()
    
    for line in lines:
        find=re.findall(r'Averaged stats: acc: ([0-9\.]+)  loss: ([0-9\.]+)',line)
        if len(find):
            #print('{}, acc: {}, loss: {}'.format(i,find[0][0],find[0][1]))
            print('{},{},{}'.format(i,find[0][0],find[0][1]))

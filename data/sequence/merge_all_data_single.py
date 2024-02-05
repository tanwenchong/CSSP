import os
import re
import json
import gzip
import numpy as np
import pandas as pd

chain='light' # 'heavy'

prefix='u{}f'.format(chain[0])

def text_regular(text,sub=r'\n'):
    return re.sub(sub,r'',text)

def get_saved_filename():
    f_record=open('antibody_data/merge_sequence/{}/{}_record.txt'.format(chain, prefix),'r',encoding='utf-8')
    file_names=f_record.readlines()
    f_record.close()
    name_set=set()
    for name_ in file_names:
        name_set.add(text_regular(name_))
    return name_set


# merge csv, csv to fasta
if 1:
    raw_dir= 'antibody_data/unpaired_{}/'.format(chain)
    print(len(os.listdir(raw_dir)))
    raw_names=os.listdir(raw_dir)
    
    name_set=get_saved_filename()
    print(len(name_set))
    
    f=open('antibody_data/merge_sequence/{}/unpaired_{}.fasta'.format(chain,chain),'a',encoding='utf-8')
    counter=0
    
    f_csv=open('antibody_data/merge_sequence/{}/unpaired_{}.csv'.format(chain,chain),'a',encoding='utf-8')
    f_csv.write('sequence,germline,cdr1,cdr2,cdr3,species,file_name\n')
    
    f_record=open('antibody_data/merge_sequence/{}/{}_record.txt'.format(chain,prefix),'a',encoding='utf-8')
    
    file_count=0
    

    for name_ in raw_names:
        if name_ in name_set:
            continue
        print('    file_count {}, processing {}'.format(file_count,name_))
        try:
            tp_=pd.read_csv(os.path.join(raw_dir,name_),compression='gzip',sep=',',skiprows=[0])
        except:
            print('#SKIP {}'.format(name_))
            continue
        aseq_=tp_['sequence_alignment_aa'].values
        gseq_=tp_['germline_alignment_aa'].values
        cdr1_=tp_['cdr1_aa'].values
        cdr2_=tp_['cdr2_aa'].values
        cdr3_=tp_['cdr3_aa'].values

        del tp_

        #species
        f_tp=gzip.open(os.path.join(raw_dir,name_),'rb')
        line=f_tp.readline().decode()
        f_tp.close()
        line=re.findall(r'""Species"": ""(.+?)"",',line)
        
        len_=len(aseq_)
        for i in range(len_):
            f.write('>{}\n{}\n'.format(counter,aseq_[i]))
            f_csv.write('{},{},{},{},{},{},{}\n'.format(aseq_[i],gseq_[i],cdr1_[i],cdr2_[i],cdr3_[i],line[0],name_))
            
            counter+=1
        f_record.write(name_+'\n')
        file_count+=1
        

    f.close()
    f_csv.close()
    f_record.close()
    
    print(counter)


import os
import re
import json
import gzip
import numpy as np
import pandas as pd

def text_regular(text,sub=r'\n'):
    return re.sub(sub,r'',text)

def get_saved_filename():
    f_record=open('antibody_data/merge_sequence/paired/upf_record.txt','r',encoding='utf-8')
    file_names=f_record.readlines()
    f_record.close()
    name_set=set()
    for name_ in file_names:
        name_set.add(text_regular(name_))
    return name_set


# merge csv, csv to fasta
if 1:
    raw_dir= 'antibody_data/paired/'
    print(len(os.listdir(raw_dir)))
    raw_names=os.listdir(raw_dir)
    
    name_set=get_saved_filename()
    print(len(name_set))
    
    f=open('antibody_data/merge_sequence/paired/paired.fasta','a',encoding='utf-8')
    counter=0
    
    f_csv=open('antibody_data/merge_sequence/paired/paired.csv','a',encoding='utf-8')
    f_csv.write('sequence_heavy,germline_heavy,cdr1_heavy,cdr2_heavy,cdr3_heavy,'
                'sequence_light,germline_light,cdr1_light,cdr2_light,cdr3_light,species,file_name\n')
    
    
    f_record=open('antibody_data/merge_sequence/paired/upf_record.txt','a',encoding='utf-8')
    
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
        aseqh_=tp_['sequence_alignment_aa_heavy'].values
        gseqh_=tp_['germline_alignment_aa_heavy'].values
        cdr1h_=tp_['cdr1_aa_heavy'].values
        cdr2h_=tp_['cdr2_aa_heavy'].values
        cdr3h_=tp_['cdr3_aa_heavy'].values
        aseql_=tp_['sequence_alignment_aa_light'].values
        gseql_=tp_['germline_alignment_aa_light'].values
        cdr1l_=tp_['cdr1_aa_light'].values
        cdr2l_=tp_['cdr2_aa_light'].values
        cdr3l_=tp_['cdr3_aa_light'].values

        del tp_

        #species
        f_tp=gzip.open(os.path.join(raw_dir,name_),'rb')
        line=f_tp.readline().decode()
        f_tp.close()
        line=re.findall(r'""Species"": ""(.+?)"",',line)
        
        len_=len(aseqh_)
        for i in range(len_):
            f.write('>{}\n{}\n'.format(counter,aseqh_[i]+'XXXXXXXXXX'+aseql_[i]))
            f_csv.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(aseqh_[i],gseqh_[i],cdr1h_[i],cdr2h_[i],cdr3h_[i],
                                                        aseql_[i],gseql_[i],cdr1l_[i],cdr2l_[i],cdr3l_[i],line[0],name_))
            
            counter+=1
        f_record.write(name_+'\n')
        file_count+=1
        

    f.close()
    f_csv.close()
    f_record.close()
    
    print(counter)

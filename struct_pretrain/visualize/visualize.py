import os
import numpy as np
import pandas as pd

import pickle
import json

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

import matplotlib.pyplot as plt

from tqdm import tqdm

chain='paired'

feature=np.load('antibody_pretrain/structure_pretrain/struct_pretrain/dedup/embedding_dedup_{}_stru_nobn24_mean.npy'.format(chain))

data_all=[]

for i in range(5):
    data_all+=json.load(open('../data/dedup/sample/{}{}.json'.format(chain,i),'r'))

label=[]
for i in range(len(data_all)):
    label.append(data_all[i]['species'])

label=np.array(label)

colors=['blue','orange','red','green','yellow','purple','gray','pink','cyan','magenta','brown']

name='tsne'

if name=='tsne':
    model=TSNE(n_components=2,init='pca',random_state=27)
    feature_tsne=model.fit_transform(feature)
    print(feature_tsne.shape)
    np.save('{}_feature_tsne.npy'.format(chain),feature_tsne)
    #feature_tsne=np.load('{}_feature_tsne.npy'.format(chain))
    species=np.unique(label)
    print(len(species))
    plt.figure(figsize=(15,15))
    for i in range(len(species)):
        spe=species[i]
        mask=(label==spe)
        plt.scatter(feature_tsne[mask][:,0],feature_tsne[mask][:,1],label=spe,c=colors[i],s=1)
    plt.legend()
    plt.savefig('tsne_{}.png'.format(chain))

if name=='pca':
    model=PCA(n_components=2,random_state=27)
    feature_pca=model.fit_transform(feature)
    print(feature_pca.shape)
    np.save('feature_pca.npy',feature_pca)
    plt.figure()
    mask1=(label==0)
    plt.scatter(feature_pca[mask1][:,0],feature_pca[mask1][:,1],label='negative',c='blue')
    mask2=(label==1)
    plt.scatter(feature_pca[mask2][:,0],feature_pca[mask2][:,1],label='positive',c='orange')
    plt.savefig('pca.png')

if name=='umap':
    model=umap.UMAP(n_components=2,min_dist=0.5,metric='correlation',random_state=27)
    feature_umap=model.fit_transform(feature)
    print(feature_umap.shape)
    plt.figure()
    mask1=(label==0)
    plt.scatter(feature_umap[mask1][:,0],feature_umap[mask1][:,1],label='negative',c='blue')
    mask2=(label==1)
    plt.scatter(feature_umap[mask2][:,0],feature_umap[mask2][:,1],label='positive',c='orange')
    plt.savefig('umap.png')
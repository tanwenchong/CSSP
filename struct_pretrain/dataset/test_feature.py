import numpy as np
from tqdm import tqdm

import torch

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

from simdesign_utils import gather_nodes, _dihedrals, _get_rbf, _orientations_coarse_gl_tuple

import matplotlib.pyplot as plt

def _full_dist(coords, mask, top_k, eps=1E-6):
    mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
    dX = torch.unsqueeze(coords,1) - torch.unsqueeze(coords,2)
    D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps) # [B, L, L]

    D_max, _ = torch.max(D, -1, keepdim=True) # D_max [B, L, 1]
    D_adjust = D + (1. - mask_2D) * (D_max+1) # [B, L, L] masked places will be large number

    D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False) # sort values in each row, return topk min
    return D_neighbors, E_idx

def _get_edge_features(E_feature, E_idx, max_sl): # [(BL-nan)*topk, 272], [2, (BL-nan)*topk]
    edge_features = torch.zeros((max_sl, max_sl, E_feature.size(1)))
    for i in range(E_idx[0][-1] + 1):
        mask_ = (E_idx[0]==i)
        edge_features[i][E_idx[1][mask_]] = E_feature[mask_]
    return edge_features.unsqueeze(0) # [1, L, L, 272]

def _get_features(seqs, coords, mask, topk=5, num_rbf=16, max_=20.0):
    '''
    seqs: [B, L]
    coords: [B, L, 4, 3]
    mask: [B, L]
    '''
    mask_bool = (mask==1)
    B, L, _,_ = coords.shape
    coords_ca = coords[:,:,1,:]
    D_neighbors, E_idx = _full_dist(coords_ca, mask, topk) # topk nearest residues, D_neighbors [B, L, topk], E_idx [B, L, topk]
    #print('D_neighbors E_idx',D_neighbors.shape,E_idx.shape)

    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1) # [B, L, topk]
    mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
    #print('mask_attend', mask_attend.shape)
    edge_mask_select = lambda x: torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])

    # sequence
    seqs = torch.masked_select(seqs, mask_bool)

    # angle & direction
    V_angles = _dihedrals(coords, 0) # [B, L, 12]
    #print('V_angles.shape',V_angles.shape)
    #V_angles = node_mask_select(V_angles) # [BL-nan, 12]
    #print('V_angles.shape nonan',V_angles.shape)
    
    V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(coords, E_idx) # [B, L, 9], [B, L, topk, 12], [B, L, topk, 4]
    #print('after _orientations_coarse_gl_tuple, V_direct.shape, E_direct.shape, E_angles.shape', V_direct.shape, E_direct.shape, E_angles.shape)
    #V_direct = node_mask_select(V_direct) # [BL-nan, 9]
    E_direct = edge_mask_select(E_direct) # [(BL-nan)*topk, 12]
    E_angles = edge_mask_select(E_angles) # [(BL-nan)*topk, 4]
    #print('after select, V_direct.shape, E_direct.shape, E_angles.shape',V_direct.shape, E_direct.shape, E_angles.shape)

    # distance
    atom_N = coords[:,:,0,:]
    atom_Ca = coords[:,:,1,:]
    atom_C = coords[:,:,2,:]
    atom_O = coords[:,:,3,:]
    b = atom_Ca - atom_N
    c = atom_C - atom_Ca
    a = torch.cross(b, c, dim=-1)

    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        rbf_ = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, num_rbf, max_=max_).squeeze() #[B, L, rbf_num]
        #print('rbf_.shape',rbf_.shape)
        #node_dist.append(node_mask_select(rbf_))
        node_dist.append(rbf_)
    #V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze() #[BL-nan, rbf_num*6]
    V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze().unsqueeze(0) #[B, L, rbf_num*6]
    
    #print('V_dist.shape',V_dist.shape)
    
    pair_lst = ['Ca-Ca', 'Ca-C', 'C-Ca', 'Ca-N', 'N-Ca', 'Ca-O', 'O-Ca', 'C-C', 'C-N', 'N-C', 'C-O', 'O-C', 'N-N', 'N-O', 'O-N', 'O-O']
    edge_dist = []
    for pair in pair_lst:
        atom1, atom2 = pair.split('-')
        rbf_ = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, num_rbf, max_=max_) # [B, L, topk, rbf_num]
        #print('rbf_.shape',rbf_.shape)
        edge_dist.append(edge_mask_select(rbf_))
    E_dist = torch.cat(tuple(edge_dist), dim=-1) # [(BL-nan)*topk, rbf_num*16]
    #print('E_dist.shape',E_dist.shape)

    #print(V_dist.shape,V_angles.shape,V_direct.shape)

    h_V = []
    h_V.append(V_dist)
    h_V.append(V_angles)
    h_V.append(V_direct)
    
    h_E = []
    h_E.append(E_dist)
    h_E.append(E_angles)
    h_E.append(E_direct)
    
    _V = torch.cat(h_V, dim=-1)
    _E = torch.cat(h_E, dim=-1)

    plt.matshow(_V.squeeze().numpy())
    plt.show()
    plt.matshow(_E.numpy())
    plt.show()
    
    # edge index
    shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
    src = shift.view(B,1,1) + E_idx
    src = torch.masked_select(src, mask_attend).view(1,-1)
    dst = shift.view(B,1,1) + torch.arange(0, L).view(1,-1,1).expand_as(mask_attend)
    dst = torch.masked_select(dst, mask_attend).view(1,-1)
    E_idx = torch.cat((dst, src), dim=0).long() # [2, (BL-nan)*topk]
    #print('E_idx',E_idx.shape)
    #print(E_idx)
    #print(E_idx[0,:])
    #print(E_idx[1,:])
    
    # 3D point
    #sparse_idx = mask.nonzero()  # index of non-zero values [B*L-nan, 2]
    #coords = coords[sparse_idx[:,0], sparse_idx[:,1], :, :]
    #batch_id = sparse_idx[:,0] # [B*L-nan,] # batch id of each sample

    edge_features = _get_edge_features(_E, E_idx, L)

    #return coords, seqs, _V, _E, E_idx, batch_id # [BL-nan, 4, 3], [BL-nan,], [BL-nan, 117], [(BL-nan)*topk, 272], [2, (BL-nan)*topk], [BL-nan, ]
    return coords, seqs, _V.squeeze(), edge_features.squeeze() # [BL-nan, 4, 3], [BL-nan,], [L, 117], [L, L, 272]


if __name__=="__main__":
    B=1
    L=5
    seqs=torch.arange(L).reshape(B,L)*1.0

    tp=torch.arange(0,500,100).reshape(5,1)[None,:,:,None].repeat((1,1,4,3))
    bias1=torch.arange(0,40,10).reshape(4,1)[None,None,:,:].repeat((1,5,1,3))
    bias2=torch.arange(3).reshape(1,3)[None,None,:,:].repeat((1,5,4,1))
    coords=(tp+bias1+bias2).float()

    coords=torch.randn(coords.shape)

    mask=torch.tensor([1,1,1,1,0]).unsqueeze(0).float()
    _,_,nfeature,efeature=_get_features(seqs, coords, mask, topk=3, num_rbf=8, max_=6.0)
    print(nfeature.shape,efeature.shape)
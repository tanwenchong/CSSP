import os
import re
import pickle

import numpy as np
from tqdm import tqdm

import torch

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

from .simdesign_utils import gather_nodes, _dihedrals, _get_rbf, _orientations_coarse_gl_tuple

def dict2feature(batch, alphabet, max_sl):
    B = len(batch)
    #lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max_sl #lengths.max()
    coords = np.zeros([B, L_max, 4, 3])
    seqs = np.zeros([B, L_max], dtype=np.int32)

    for i, b in enumerate(batch):
        coord_ = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1) # [l, 4, 3]
        
        l = len(b['seq'])
        coord_pad = np.pad(coord_, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [L, 4, 3]
        coords[i,:,:,:] = coord_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        seqs[i, :l] = indices

    mask = np.isfinite(np.sum(coords,(2,3))).astype(np.float32) # mask of valid atoms. set as True when all the atoms exist [B, L]
    numbers = np.sum(mask, axis=1).astype(int) # number of valid atoms. [B, ]
    seqs_new = np.zeros_like(seqs)
    coords_new = np.zeros_like(coords)+np.nan
    for i, n in enumerate(numbers):
        coords_new[i,:n,::] = coords[i][mask[i]==1]
        seqs_new[i,:n] = seqs[i][mask[i]==1]
        
    # only keep residues with all atoms valid
    coords = coords_new
    seqs = seqs_new
    isnan = np.isnan(coords)
    mask = np.isfinite(np.sum(coords,(2,3))).astype(np.float32)
    coords[isnan] = 0.
    # Conversion
    seqs = torch.from_numpy(seqs).to(dtype=torch.long)
    coords = torch.from_numpy(coords).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)

    #print(seqs.shape, coords.shape, mask.shape)
    
    return seqs, coords, mask

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

def _get_features(seqs, coords, mask, topk=5, num_rbf_node=8, num_rbf_edge=16):
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
        rbf_ = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, num_rbf_node).squeeze() #[B, L, rbf_num]
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
        rbf_ = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, num_rbf_edge) # [B, L, topk, rbf_num]
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
    #print(_E)
    #print(edge_features)

    #return coords, seqs, _V, _E, E_idx, batch_id # [BL-nan, 4, 3], [BL-nan,], [BL-nan, 117], [(BL-nan)*topk, 272], [2, (BL-nan)*topk], [BL-nan, ]
    return coords, seqs, _V.squeeze(), edge_features.squeeze() # [BL-nan, 4, 3], [BL-nan,], [L, 117], [L, L, 272]

def _get_features0(seqs, coords, mask, topk=5, num_rbf=16, max_sl = 256):
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
    node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

    # sequence
    seqs = torch.masked_select(seqs, mask_bool)

    # angle & direction
    V_angles = _dihedrals(coords, 0) # [B, L, 12]
    #print('V_angles.shape',V_angles.shape)
    V_angles = node_mask_select(V_angles) # [BL-nan, 12]
    #print('V_angles.shape nonan',V_angles.shape)
    
    V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(coords, E_idx) # [B, L, 9], [B, L, topk, 12], [B, L, topk, 4]
    #print('after _orientations_coarse_gl_tuple, V_direct.shape, E_direct.shape, E_angles.shape', V_direct.shape, E_direct.shape, E_angles.shape)
    V_direct = node_mask_select(V_direct) # [BL-nan, 9]
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
        rbf_ = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, num_rbf).squeeze() #[B, L, rbf_num]
        #print('rbf_.shape',rbf_.shape)
        node_dist.append(node_mask_select(rbf_))
    V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze() #[BL-nan, rbf_num*6]
    
    #print('V_dist.shape',V_dist.shape)
    
    pair_lst = ['Ca-Ca', 'Ca-C', 'C-Ca', 'Ca-N', 'N-Ca', 'Ca-O', 'O-Ca', 'C-C', 'C-N', 'N-C', 'C-O', 'O-C', 'N-N', 'N-O', 'O-N', 'O-O']
    edge_dist = []
    for pair in pair_lst:
        atom1, atom2 = pair.split('-')
        rbf_ = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, num_rbf) # [B, L, topk, rbf_num]
        #print('rbf_.shape',rbf_.shape)
        edge_dist.append(edge_mask_select(rbf_))
    E_dist = torch.cat(tuple(edge_dist), dim=-1) # [(BL-nan)*topk, rbf_num*16]
    #print('E_dist.shape',E_dist.shape)

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
    sparse_idx = mask.nonzero()  # index of non-zero values [B*L-nan, 2]
    coords = coords[sparse_idx[:,0], sparse_idx[:,1], :, :]
    batch_id = sparse_idx[:,0] # [B*L-nan,] # batch id of each sample

    return coords, seqs, _V, _E, E_idx, batch_id # [BL-nan, 4, 3], [BL-nan,], [BL-nan, 117], [(BL-nan)*topk, 272], [2, (BL-nan)*topk], [BL-nan, ]



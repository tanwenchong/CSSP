import random
import numpy as np


def _find_range_idx(seq, target):
    if target is None:
        return -1, -1
    slen,tlen=len(seq),len(target)
    if tlen==0:
        return -1,-1

    i=0
    while i<=slen-tlen:
        j=0
        if seq[i]==target[j]:
            while (j<tlen) and (seq[i]==target[j]):
                i+=1
                j+=1
            if j==tlen:
                return i-tlen,i-1 # index of the first and the last token
            else:
                i-=j-1
                continue
        else:
            i+=1
    return -1,-1

def get_focused_span_mask(label_ids, seq_cdr, mask_prob):
    # SEQ [PAD]
    
    ismask=np.zeros(label_ids.shape)
    cdr_keys=seq_cdr.keys()

    # mask cdr
    masked_len=0
    for range_ in cdr_keys:
        bidx,eidx=_find_range_idx(label_ids,seq_cdr[range_])
        if bidx>=0:
            len_=eidx-bidx
            bidx+=random.randint(0,len_//3) # mask part of cdr
            eidx-=random.randint(0,len_//3)
            ismask[bidx:eidx+1]=1
            masked_len+=eidx+1-bidx
            
    # mask rest
    mask_len = len(label_ids)*mask_prob - masked_len
    sidxs = iter(np.random.permutation(len(label_ids)))
    for trial in range(3):
        slens = np.random.poisson(3, len(label_ids))
        slens[slens < 3] = 3
        slens[slens > 8] = 8
        slens = slens[slens.cumsum() < mask_len]
        if len(slens) != 0:
            break
    for slen in slens:
        for trial in range(3):
            sid = next(sidxs)
            lid = sid - 1      # do not merge two spans
            rid = sid + slen   # do not merge two spans
            if lid >= 0 and rid < len(label_ids) and ismask[lid] != 1 and ismask[rid] != 1:
                ismask[sid: sid + slen] = 1
                break
    
    return ismask.astype(bool)

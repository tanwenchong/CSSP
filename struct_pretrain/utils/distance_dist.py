import os
import numpy as np
import matplotlib.pyplot as plt

edges=[]
nodes=[]
files=os.listdir('./')
for file in files:
    if file[:5]=='edist':
        edges.append(np.load(file).flatten())
    elif file[:5]=='ndist':
        nodes.append(np.load(file).flatten())
edges=np.concatenate(edges,axis=0)
nodes=np.concatenate(nodes,axis=0)

plt.figure()
plt.hist(edges)
plt.savefig('edge_hist.png')
plt.figure()
plt.hist(nodes)
plt.savefig('node_hist.png')
'''
Model consistency: 
    Run with synthetic data K - 2 .. 8 
    Synthetic alpha = 0.2 
    N = 100 
    K = 3 
    D = 2 
    CV = 5 
    seed = 42
    Sample_size = 1 
    Lr = 0.01 (default) 
    Iterations = 10,000 
'''

from turtle import color
from src.data.synthetic_data import main
from src.models.train_DRRAA_module import DRRAA
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.models.calcNMI import calcNMI
import matplotlib as mpl
import scipy.stats as st

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return
setup_mpl()

num_init = 5 
iter = 10
#Get synthetic data and convert to edge list
d=2
archs = [2, 3, 4, 5, 6, 7, 8]
NMIs = []
for k in archs:
    adj_m, z, A, Z_true, beta = main(alpha=.2, k=k, dim=d, nsamples=100, rand=False) #z is cmap
    Graph = nx.from_numpy_matrix(adj_m.numpy())
    temp = [x for x in nx.generate_edgelist(Graph, data=False)]
    edge_list = np.zeros((2, len(temp)))
    for i in range(len(temp)): 
        edge_list[0, i] = temp[i].split()[0]
        edge_list[1, i] = temp[i].split()[1]
    #edge_list = torch.FloatTensor(edge_list).long()

    NMI = []

    for i in range(num_init):
        model = DRRAA(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list')
        model.train(iterations=iter, LR=0.01)

        Z = F.softmax(model.Z, dim=0)
        G = F.sigmoid(model.Gate)
        C = (Z.T * G) / (Z.T * G).sum(0)

        u, sigma, v = torch.svd(model.A) # Decomposition of A.
        r = torch.matmul(torch.diag(sigma), v.T)
        embeddings = torch.matmul(r, torch.matmul(torch.matmul(Z, C), Z)).T
        archetypes = torch.matmul(r, torch.matmul(Z, C))
        NMI.append(calcNMI(Z, Z_true).item())
    NMIs.append(NMI)

fig, ax = plt.subplots(figsize=(10,5), dpi=500)
ax.boxplot(NMIs, color='#C4000D')
ax.set_ylabel("NMI")
ax.set_xlabel("k: number of archetypes")
ax.grid(alpha=.3)
plt.savefig("consistency.png", dpi=500)
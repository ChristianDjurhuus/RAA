

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

seed = 100
torch.random.manual_seed(seed)
np.random.seed(seed)
true_k = 8
true_alpha = 0.05
adj_m, z, A, Z_true, beta, partition_cmap = main(alpha=true_alpha, k=true_k, dim=2, nsamples=1000, rand=True) #z is cmap
G = nx.from_numpy_matrix(adj_m.numpy())
temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)): 
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]


NMIs = []
for iter in [100000]:#[75000, 100000]:#[10000, 20000, 30000, 40000, 50000, 75000, 100000]:
    raa = DRRAA(k=8,
                d=2, 
                sample_size=.5,
                data=edge_list,
                link_pred=False)

    raa.train(iterations=iter, LR=0.01, print_loss=True, patience=0.95)
    raa.plot_latent_and_loss(iterations=iter, cmap=partition_cmap, file_name="embedding_and_loss_complex_patience.png")
    Z = F.softmax(raa.Z, dim=0)
    G = F.sigmoid(raa.Gate)
    C = (Z.T * G) / (Z.T * G).sum(0)
    u, sigma, v = torch.svd(raa.A) # Decomposition of A.
    r = torch.matmul(torch.diag(sigma), v.T)
    embeddings = torch.matmul(r, torch.matmul(torch.matmul(Z, C), Z)).T
    archetypes = torch.matmul(r, torch.matmul(Z, C))

    #Calculate NMI between embeddings
    print(f'The NMI between z and z_hat is {calcNMI(Z, Z_true)}')
    NMIs.append((iter, calcNMI(Z, Z_true)))
print(NMIs)


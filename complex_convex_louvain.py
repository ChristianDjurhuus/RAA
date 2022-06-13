

from turtle import color
from src.data.synthetic_data import main
from src.models.train_DRRAA_module import DRRAA
from src.models.train_KAA_module import KAA
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.models.calcNMI import calcNMI
import matplotlib as mpl
from src.data.synthetic_data import truncate_colormap

seed = 100
torch.random.manual_seed(seed)
np.random.seed(seed)
NMIs = []
AUCs = []
true_k = 7
true_alpha = 0.05
adj_m, z, A, Z_true, beta, partition_cmap = main(alpha=true_alpha, k=true_k, dim=2, nsamples=1000, rand=True) #z is cmap
G = nx.from_numpy_matrix(adj_m.numpy())
temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)): 
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

#15000, 0.5, 0.065
for iter in [15000]:#[10000, 20000, 30000, 40000, 50000]:#[50000,60000,100000]:#[75000, 100000]:#[10000, 20000, 30000, 40000, 50000, 75000, 100000]:
    
    kaa = KAA(k=true_k,
            data=adj_m.numpy(),
            data_type="adjacency matrix")
    kaa.train(iterations=1000)
    raa = DRRAA(k=true_k,
                d=2, 
                sample_size=0.5,
                data=edge_list,
                link_pred=False,
                init_Z=kaa.S.detach())
    #raa2 = DRRAA(k=8,
    #        d=2,
    #        sample_size=.3,
    #        data=edge_list,
    #        link_pred=True)

    raa.train(iterations=iter, LR=0.065, print_loss=True, scheduling=False, early_stopping=0.8)
    #raa2.train(iterations=iter, LR=0.03, print_loss=False, scheduling=True, early_stopping=0.8)
    #auc, _, _ = raa2.link_prediction()
    #raa.plot_latent_and_loss(iterations=iter, c=z, file_name=f"embedding_and_loss_complex_patience_{iter}_rand.png")
    Z = F.softmax(raa.Z, dim=0)
    Gate = F.sigmoid(raa.Gate)
    C = (Z.T * Gate) / (Z.T * Gate).sum(0)
    u, sigma, v = torch.svd(raa.A) # Decomposition of A.
    r = torch.matmul(torch.diag(sigma), v.T)
    embeddings = torch.matmul(r, torch.matmul(torch.matmul(Z, C), Z)).T
    archetypes = torch.matmul(r, torch.matmul(Z, C))

    fig, ax1 = plt.subplots(dpi=200)
    cmap = plt.get_cmap('RdPu')
    cmap = truncate_colormap(cmap, 0.2, 1)
    pos_map = dict(list(zip(G.nodes(), list(embeddings.detach().numpy()))))
    org_pos_map = dict(list(zip(G.nodes(), list((A@Z_true).T.detach().numpy()))))
    nx.draw_networkx_nodes(G, pos=pos_map, ax = ax1, node_color=z, alpha=1, node_size=[v for v in dict(G.degree).values()], cmap=cmap)
    nx.draw_networkx_edges(G, pos=pos_map, ax = ax1, alpha=.1)
    plt.savefig("complex_convex_reg.png", dpi=200)
    fig, ax2 = plt.subplots(dpi=200)
    nx.draw_networkx_nodes(G, pos=org_pos_map, ax=ax2, node_color=z, alpha=1, node_size=[v for v in dict(G.degree).values()], cmap=cmap)
    nx.draw_networkx_edges(G, pos=org_pos_map, ax=ax2, alpha=.1)
    plt.savefig("complex_convex_org.png", dpi=200)
    #Calculate NMI between embeddings
    print(f'The NMI between z and z_hat is {calcNMI(Z, Z_true)}')
    NMIs.append((i*3, calcNMI(Z, Z_true)))
    #AUCs.append((iter, auc))
    plt.close()
    raa.order_adjacency_matrix(filename="complex_convex_ordered_adj.png")
print(NMIs)
#print(AUCs)


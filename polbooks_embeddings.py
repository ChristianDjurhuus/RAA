from typing_extensions import Self
import torch
import torch.nn.functional as F
import numpy as np
from src.models.train_DRRAA_module import DRRAA
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import matplotlib.patches as mpatches
from src.visualization.interactive_plot import get_Plotly_netw
import igraph as ig
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)


dataset = 'polbooks'
org_data =  torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_i.txt")).long()
org_data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_j.txt")).long()
metadata = nx.read_gml('data/raw/polbooks/polbooks.gml')
metadata = list(nx.get_node_attributes(metadata,'value').values())
k=3
d=2
iter = 1000
model = DRRAA(data = org_data,
            data_2 = org_data2,
            k = k,
            d = d,
            data_type = "sparse",
            sample_size=1, # Set sampling procentage size
            seed_init = 10,
            link_pred = False,
            )

model.train(iterations=iter, LR=0.1, print_loss=True)
latent_z = model.Softmax(model.latent_z1)
G = torch.sigmoid(model.Gate)
C = ((model.latent_z * G) / (model.latent_z * G).sum(0))
AZC = model.A@(latent_z.transpose(0,1)@C)
embeddings = (AZC@(latent_z.transpose(0,1))).transpose(0,1).detach().numpy()
archetypes = AZC.detach().numpy()
import matplotlib
import matplotlib.lines as mlines
import matplotlib.colors as mcol
import matplotlib.cm as cm

f, ax = plt.subplots()
map = {'c':'red','l':'blue','n':'green'}
ax.scatter(embeddings[:, 0], embeddings[:, 1], c=[map[i] for i in metadata],
           label="Node embeddings", marker='o')
ax.scatter(archetypes[0,0], archetypes[1,0], edgecolors='black', marker='^',s=100, c='green', label="Archetypes")
ax.scatter(archetypes[0,1], archetypes[1,1], edgecolors='black', marker='^',s=100, c='red', label="Archetypes")
ax.scatter(archetypes[0,2], archetypes[1,2], edgecolors='black', marker='^',s=100, c='blue', label="Archetypes")
dem = mlines.Line2D([], [], color='blue', marker='^', linestyle='None',
                          markersize=10, label='Democratic')
rep = mlines.Line2D([], [], color='red', marker='^', linestyle='None',
                          markersize=10, label='Republican')
ind = mlines.Line2D([], [], color='green', marker='^', linestyle='None',
                          markersize=10, label='Neutral')

plt.legend(handles=[dem, rep, ind])
plt.show()
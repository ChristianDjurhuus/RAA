from typing_extensions import Self
import torch
import torch.nn.functional as F
import numpy as np
from src.models.train_DRRAA_module import DRRAA
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from src.visualization.interactive_plot import get_Plotly_netw
import igraph as ig

torch.manual_seed(42)


dataset = '2020_congress_uni'
org_data =  torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_i.txt")).long()
org_data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_j.txt")).long()
values = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_values.txt")).long()
metadata = pd.read_csv("data/raw/2020_congress/2020_congress_metadata.csv")

k=3
d=2
iter = 1000
model = DRRAA(data = org_data,
            data_2 = org_data2,
            k = k,
            d = d,
            data_type = "sparse",
            sample_size=1,
            seed_init = 0,
            link_pred = False,
            values=values) # Set sampling procentage size

model.train(iterations=iter, LR=0.1, print_loss=True)
latent_z = model.Softmax(model.latent_z1)
G = torch.sigmoid(model.Gate)
C = ((model.latent_z * G) / (model.latent_z * G).sum(0))
AZC = model.A@(latent_z.transpose(0,1)@C)
embeddings = (AZC@(latent_z.transpose(0,1))).transpose(0,1).detach().numpy()
archetypes = AZC.detach().numpy()


map = {'Democratic':'darkblue', 'Republican':'firebrick', 'Independent':'darkgreen'}
plt.scatter(embeddings[:,0], embeddings[:,1], c=metadata['Party'].replace(map), label="Node embeddings")
plt.scatter(archetypes[0,:], archetypes[1,:], marker='^', c='black', label="Archetypes")
red_patch = mpatches.Patch(color='red', label='Republican')
blue_patch = mpatches.Patch(color='blue', label='Democratic')
green_patch = mpatches.Patch(color='green', label='Independent')
plt.legend(handles=[red_patch, blue_patch, green_patch])
plt.show()

attrs = {'color' : metadata['Party'].replace(map).values.tolist(),
        'state' : metadata['State'].values.tolist(),
        'party' : metadata['Party'].values.tolist(),
        'name' : metadata['Name'].values.tolist()}

graph = ig.Graph(n=model.N,
                 edges = [list(x) for x in zip(org_data.tolist(),org_data2.tolist())],
                 edge_attrs = {'weight':values.tolist()},
                 vertex_attrs = attrs)
layout = embeddings
fig = get_Plotly_netw(graph, layout)
fig.write_html("congress_network.html")
fig.show()
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
from graspologic.plot import adjplot, matrixplot

dataset = '2020_congress_uni'
org_data =  torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_i.txt")).long()
org_data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_j.txt")).long()
values = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_values.txt"))
metadata = pd.read_csv("data/raw/2020_congress/2020_congress_metadata.csv")

k=2
d=2
iter = 1000
model = DRRAA(data = org_data,
            data_2 = org_data2,
            k = k,
            d = d,
            data_type = "sparse",
            sample_size=1,
            link_pred = False,
            values=values) # Set sampling procentage size

model.train(iterations=iter, LR=0.1, print_loss=True)
latent_z = model.Softmax(model.latent_z1)
G = torch.sigmoid(model.Gate)
C = ((latent_z * G) / (latent_z * G).sum(0))
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

z_idx=latent_z.detach().numpy().argmax(1)
w_idx=latent_z.detach().numpy().argmax(1)

f_z=z_idx.argsort()
f_w=w_idx.argsort()


edge_list = torch.stack((org_data, org_data2))
X = torch.sparse_coo_tensor(edge_list, values.numpy(), (model.N, model.N), device=model.device)
X = X.to_dense()
i_lower = np.tril_indices(X.shape[0], -1)
X[i_lower] = X.T[i_lower]

D = X[:, f_w][f_z]

fig, ax = plt.subplots(dpi=100)
polmap = {'Democratic':'Dem', 'Republican':'Rep', 'Independent':'Ind'}
meta = pd.DataFrame(data={'Archetype':z_idx, 'Party':metadata['Party'].replace(polmap).values.tolist()})
gridline_kws = dict(color="black", linestyle="--", alpha=0.7, linewidth=1)
adjplot(data=X.numpy(),
        ax=ax,
        meta=meta,
        plot_type='heatmap',
        group=['Archetype', 'Party'],
        sizes=(1,30),
        gridline_kws=gridline_kws)

plt.show()
plt.savefig("vote_rec_adj_m.png", dpi=100)


'''attrs = {'color' : metadata['Party'].replace(map).values.tolist(),
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
fig.show()'''
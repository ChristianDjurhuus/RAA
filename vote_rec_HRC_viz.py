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
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)


dataset = '2020_congress_uni'
org_data =  torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_i.txt")).long()
org_data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_j.txt")).long()
values = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_values.txt")).long()
metadata = pd.read_csv("data/raw/2020_congress/2020_congress_metadata.csv")
HRC = pd.read_csv('116th_Congress_HRC_scores.csv', sep=';')

metadata['116'], metadata['115'], metadata['114'] = np.full([442], np.nan), np.full([442], np.nan), np.full([442], np.nan)

for idx,name in enumerate(metadata['Name']):
    for idx2, name2 in enumerate(HRC['Name']):
        if (name==name2) and (metadata['State'][idx]==HRC['State'][idx2]):
            metadata['116'][idx] = HRC['116'][idx2]
            metadata['115'][idx] = HRC['115'][idx2]
            metadata['114'][idx] = HRC['114'][idx2]

k=2
d=2
iter = 1000
model = DRRAA(data = org_data,
            data_2 = org_data2,
            k = k,
            d = d,
            data_type = "sparse",
            sample_size=0.5, # Set sampling procentage size
            seed_init = 1,
            link_pred = False,
            values=values)

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
vmin, vmax = 0,100
cm1 = mcol.LinearSegmentedColormap.from_list("pol",["r","b"])
cnorm = mcol.Normalize(vmin=vmin,vmax=vmax)
cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
cpick.set_array([])
for i in metadata.sort_values('Party')['Unnamed: 0'].index:
    if metadata['Party'][i]=='Republican':
        ax.scatter(embeddings[i, 0], embeddings[i, 1], c=cpick.to_rgba(metadata['116'][i]), vmin=vmin,vmax=vmax,
                   label="Node embeddings", marker='s')
    if metadata['Party'][i]=='Democratic':
        ax.scatter(embeddings[i, 0], embeddings[i, 1], c=cpick.to_rgba(metadata['116'][i]), vmin=vmin,vmax=vmax,
                   label="Node embeddings", marker='o')
    if metadata['Party'][i]=='Independent':
        ax.scatter(embeddings[i, 0], embeddings[i, 1], c=cpick.to_rgba(metadata['116'][i]), vmin=vmin,vmax=vmax,
                   label="Node embeddings", marker='D')
f.colorbar(cpick,label="HRC Score")
ax.scatter(archetypes[0,0], archetypes[1,0], edgecolors='black', marker='^',s=100, c='blue', label="Archetypes")
ax.scatter(archetypes[0,1], archetypes[1,1], edgecolors='black', marker='^',s=100, c='red', label="Archetypes")
dem = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='Democratic')
rep = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=10, label='Republican')
ind = mlines.Line2D([], [], color='green', marker='D', linestyle='None',
                          markersize=10, label='Independent')

plt.legend(handles=[dem, rep, ind])
plt.show()
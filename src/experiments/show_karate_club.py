from src.models.train_DRRAA_module import DRRAA
import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as st

seed = 1986
torch.random.manual_seed(seed)
np.random.seed(seed)

#import data
karate_club = nx.karate_club_graph()

#define model
k = 5
RAA = DRRAA(d=2, k=k, data=karate_club, data_type='networkx',link_pred=False, sample_size=1)
iter=10000
RAA.train(iterations=iter)

#get colorlist
color_list = ["303638","f0c808","5d4b20","469374","9341b3","e3427d","e68653","ebe0b0","edfbba","ffadad","ffd6a5","fdffb6","caffbf","9bf6ff","a0c4ff","bdb2ff","ffc6ff","fffffc"]
color_list = ["#"+i.lower() for i in color_list]


color_map = [color_list[4] if karate_club.nodes[i]['club'] == 'Mr. Hi' else color_list[5] for i in karate_club.nodes()]


np.random.seed(seed)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10), dpi=500)
d = dict(karate_club.degree)
embeddings, archetypes = RAA.get_embeddings()
pos_map = dict(list(zip(karate_club.nodes(), list(embeddings))))
nx.draw(karate_club, ax = ax1, node_color=color_map, alpha=.8, node_size=[10*v for v in d.values()], edge_color = (0.75, 0.75, 0.75))
nx.draw(karate_club, ax= ax2, node_color=color_map, alpha=.8, pos=pos_map, node_size=[10*v for v in d.values()], edge_color = (0.75, 0.75, 0.75) )
ax2.scatter(archetypes[0, :], archetypes[1, :], marker='^', c='black', label="Archetypes", s=80)
ax2.legend()
ax1.set_title('Networkx\'s Spring Layout')
ax2.set_title('RAA\'s Embeddings')
plt.savefig(f'show_karate_club_{k}.png',dpi=500)
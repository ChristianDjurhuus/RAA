import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.models.train_DRRAA_module import DRRAA
import torch
import numpy as np
import netwulf as nw

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return
setup_mpl()

#import data
G = nx.read_gml("data/raw/dolphins/dolphins.gml")
#G = nx.read_gml("data/raw/dolphin/dolphins.gml")
G = G.to_undirected()

if nx.number_connected_components(G) > 1:
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
label_map = {x: i for i, x in enumerate(G.nodes)}
G = nx.relabel_nodes(G, label_map)

#kvals = [2, 3, 5, 10, 15]
kvals = [3]
for k in kvals:
    #define model
    RAA = DRRAA(d=2, k=k, data=G, data_type='networkx',link_pred=True, sample_size=0.5)
    iter=10000
    RAA.train(iterations=iter)
    raa_auc, fpr, tpr = RAA.link_prediction()

    #get colorlist
    color_list = ["303638","f0c808","5d4b20","469374","9341b3","e3427d","e68653","ebe0b0","edfbba","ffadad","ffd6a5","fdffb6","caffbf","9bf6ff","a0c4ff","bdb2ff","ffc6ff","fffffc"]
    color_list = ["#"+i.lower() for i in color_list]
    color_map = [color_list[14] if G.nodes[i]['value'] == 0 else color_list[5] for i in G.nodes()]

    #draw graph
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,5), dpi=500)
    d = dict(G.degree)
    embeddings, archetypes = RAA.get_embeddings()
    pos_map = dict(list(zip(G.nodes(), list(embeddings))))
    nx_pos_map = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=nx_pos_map, ax = ax1, node_color=color_map, alpha=.9, node_size=[v for v in d.values()])
    nx.draw_networkx_edges(G, pos=nx_pos_map, ax = ax1, alpha=.2)
    nx.draw_networkx_nodes(G, pos=pos_map, ax=ax2, node_color=color_map, alpha=.9, node_size=[v for v in d.values()])
    nx.draw_networkx_edges(G, pos=pos_map, ax=ax2, alpha=.1)

    ax2.scatter(archetypes[0, :], archetypes[1, :], marker='^', c='black', label="Archetypes", s=80)
    ax2.legend()
    ax1.set_title('Networkx\'s Spring Layout')
    ax2.set_title('RAA\'s Embeddings')
    ax3.plot(fpr, tpr, '#C4000D', label='AUC = %0.2f' % raa_auc)
    ax3.plot([0, 1], [0, 1], 'b--', label='random')
    ax3.legend(loc='lower right')
    ax3.set_xlabel("False positive rate")
    ax3.set_ylabel("True positive rate")
    ax3.set_title("AUC")
    ax4.plot([i for i in range(1,iter+1)], RAA.losses, c="#C4000D")
    ax4.set_title("Loss")
    #plt.savefig(f"show_polblogs_{k}.png", dpi=500)
    #plt.show()

    RAA.KNeighborsClassifier([])




from turtle import color
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.models.train_DRRAA_module import DRRAA
from src.models.train_KAA_module import KAA
from src.models.train_LSM_module import LSM
from src.models.train_LSM_module import LSMAA
import torch
import numpy as np
import scipy.stats as st

seed = 1
torch.random.manual_seed(seed)
np.random.seed(seed)

#import data
G = nx.read_gml("data/raw/polbooks/polbooks.gml")
G = G.to_undirected()

if nx.number_connected_components(G) > 1:
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
label_map = {x: i for i, x in enumerate(G.nodes)}
reverse_label_map = {i: x for x,i in label_map.items()}
G = nx.relabel_nodes(G, label_map)

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)):
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]


num_init = 5
best_loss_raa = 1e16
best_loss_lsm = 1e16
seed_split = 1
seed_init = 1
iter = 5000

LSM_aucs = []
lsm_accs_knn = []

for _ in range(num_init):
    lsm = LSM(d=2,
            sample_size=1,
            data = edge_list,
            data_type="edge list",
            link_pred=True,
            test_size=0.3,
            seed_split = seed_split,
            seed_init=seed_init
            )
    lsm.train(iterations=iter+1000, LR=0.01, print_loss=False)
    #Saving best model
    if np.mean(lsm.losses[-100:]) < best_loss_lsm:
        best_lsm = lsm
        best_loss_lsm = np.mean(lsm.losses[-100:])
    lsm_auc, _, _ = lsm.link_prediction()
    lsm_knn_mu, lsm_knn_ci, lsm_knn_std = lsm.KNeighborsClassifier(list(nx.get_node_attributes(G, "value").values()))
    lsm_accs_knn.append(lsm_knn_mu)

    LSM_aucs.append(lsm_auc)
    seed_init += 1


RAA_avgAUCs = {}
RAA_conf_AUCs = {}

RAA_avg_accs_knn = {}
RAA_conf_accs_knn = {}

kvals = [2,3,4,5,6,7,8]

for k in kvals:
    #seed_init = 1
    kaa = KAA(k=k, 
              data=edge_list,
              data_type="edge list",
              link_pred=True,
              test_size=0.3,
              seed_split=seed_split,
              seed_init=seed_init
              )
    kaa.train(iterations=1000)
    RAA_aucs = []
    RAA_accs_knn = []
    RAA_accs_lg = []

    for _ in range(num_init):
        #define model
        RAA = DRRAA(d=2, 
                    k=k,
                    data=edge_list,
                    data_type='edge list',
                    link_pred=True,
                    test_size=0.2,
                    sample_size=1,
                    seed_split=seed_split,
                    seed_init=seed_init,
                    init_Z=kaa.S.detach())

        RAA.train(iterations=iter, LR=0.1, print_loss=False, scheduling=False, early_stopping=0.8)
        #Saving best model
        if np.mean(RAA.losses[-100:]) < best_loss_raa:
            best_loss_raa = np.mean(RAA.losses[-100:])
            best_raa = RAA
        
        raa_auc, _, _ = RAA.link_prediction()
        raa_knn_mu, raa_knn_ci, raa_knn_std = RAA.KNeighborsClassifier(list(nx.get_node_attributes(G, "value").values()))

        RAA_accs_knn.append(raa_knn_mu)
        RAA_aucs.append(raa_auc)
        seed_init += 1
    
    RAA_avgAUCs[k] = np.mean(RAA_aucs)
    RAA_conf_AUCs[k] = st.t.interval(alpha=0.95, df=len(RAA_aucs)-1, 
                    loc=np.mean(RAA_aucs), 
                    scale=st.sem(RAA_aucs))

    RAA_avg_accs_knn[k] = np.mean(RAA_accs_knn)
    RAA_conf_accs_knn[k] = st.t.interval(alpha=0.95, df=len(RAA_accs_knn)-1, 
                    loc=np.mean(RAA_accs_knn), 
                    scale=st.sem(RAA_accs_knn))


fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(kvals, list(RAA_avgAUCs.values()), '-o', label="RAA", color='#e3427d')
ax.plot(kvals, [x for (x,y) in RAA_conf_AUCs.values()], '--', color='#e3427d')
ax.plot(kvals, [y for (x,y) in RAA_conf_AUCs.values()], '--', color='#e3427d')
ax.fill_between(kvals,
                 y1 = [x for (x,y) in RAA_conf_AUCs.values()],
                 y2 = [y for (x,y) in RAA_conf_AUCs.values()], color='#e3427d', alpha=0.2)

avg_lsm_aucs = [0]*len(kvals) + np.mean(LSM_aucs)
conf_lsm_aucs = st.t.interval(alpha=0.95, df=len(LSM_aucs)-1, 
                        loc=np.mean(LSM_aucs), 
                        scale=st.sem(LSM_aucs))

ax.plot(kvals, avg_lsm_aucs, '-o', label="LDM", color="#e68653")
ax.fill_between(kvals,
                 y1 = [0]*len(kvals) + conf_lsm_aucs[0],
                 y2 = [0]*len(kvals) + conf_lsm_aucs[1],
                 color='#e68653', alpha=0.2)
ax.plot(kvals, [0]*len(kvals) + conf_lsm_aucs[0], '--', color='#e68653')
ax.plot(kvals,  [0]*len(kvals) + conf_lsm_aucs[1], '--', color='#e68653')
ax.legend()
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("AUC")
plt.savefig("polbooks_aucs.png")


fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(kvals, list(RAA_avg_accs_knn.values()), '-o', label="RAA - KNN", color='C1')
ax.plot(kvals, [x for (x,y) in RAA_conf_accs_knn.values()], '--', color='C1')
ax.plot(kvals, [y for (x,y) in RAA_conf_accs_knn.values()], '--', color='C1')
ax.fill_between(kvals,
                 y1 = [x for (x,y) in RAA_conf_accs_knn.values()],
                 y2 = [y for (x,y) in RAA_conf_accs_knn.values()], color='C1', alpha=0.2)


#LSM
avg_lsm_accs_knn = [0]*len(kvals) + np.mean(lsm_accs_knn)
conf_lsm_accs_knn = st.t.interval(alpha=0.95, df=len(lsm_accs_knn)-1, 
                        loc=np.mean(lsm_accs_knn), 
                        scale=st.sem(lsm_accs_knn))

ax.plot(kvals, avg_lsm_accs_knn, '-o', label="LDM - KNN", color="C3")
ax.fill_between(kvals,
                 y1 = [0]*len(kvals) + conf_lsm_accs_knn[0],
                 y2 = [0]*len(kvals) + conf_lsm_accs_knn[1],
                 color='C3', alpha=0.2)
ax.plot(kvals, [0]*len(kvals) + conf_lsm_accs_knn[0], '--', color='C3')
ax.plot(kvals,  [0]*len(kvals) + conf_lsm_accs_knn[1], '--', color='C3')
ax.legend()
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("Accuracy")
plt.savefig("polbooks_accs.png")


#get colorlist
color_list = ["303638","f0c808","5d4b20","469374","9341b3","e3427d","e68653","ebe0b0","edfbba","ffadad","ffd6a5","fdffb6","caffbf","9bf6ff","a0c4ff","bdb2ff","ffc6ff","fffffc"]
color_list = ["#"+i.lower() for i in color_list]
#color_map = [color_list[14] if G.nodes[i]['value'] == 'l' else color_list[5] for i in G.nodes()]
color_map = []
for i in G.nodes():
    if G.nodes[i]['value'] == 'l':
        color_map.append(color_list[14])
    elif G.nodes[i]['value'] == 'c':
        color_map.append(color_list[5])
    else:
        color_map.append(color_list[1])


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10), dpi=100)
d = dict(G.degree)
raa_embeddings, raa_archetypes = best_raa.get_embeddings()
lsm_embeddings = best_lsm.get_embeddings()[0]
pos_map_raa = dict(list(zip(G.nodes(), list(raa_embeddings))))
pos_map_lsm = dict(list(zip(G.nodes(), list(lsm_embeddings))))
nx.draw_networkx_nodes(G, pos=pos_map_raa, ax = ax1, node_color=color_map, alpha=.9, node_size=[v for v in d.values()])
nx.draw_networkx_edges(G, pos=pos_map_raa, ax = ax1, alpha=.1)
nx.draw_networkx_nodes(G, pos=pos_map_lsm, ax=ax2, node_color=color_map, alpha=.9, node_size=[v for v in d.values()])
nx.draw_networkx_edges(G, pos=pos_map_lsm, ax=ax2, alpha=.1)
archetypal_nodes = best_raa.archetypal_nodes()
ax1.scatter(raa_archetypes[0, :], raa_archetypes[1, :], marker='^', c='black', label="Archetypes", s=80)
for i in archetypal_nodes:
    ax1.annotate(reverse_label_map[int(i)], 
                    xy=(raa_embeddings[int(i),:]),
                    xytext=(raa_embeddings[int(i),:])*1.005,
                    bbox=dict(boxstyle="round4",
                    fc=color_map[int(i)],
                    ec="black",
                    lw=2),
                    arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3,rad=-0.2",
                                fc="w"))
ax1.legend()
ax2.set_title('LSM\'s Embeddings')
ax1.set_title('RAA\'s Embeddings')
plt.savefig("polbooks_embeddings_test.png")


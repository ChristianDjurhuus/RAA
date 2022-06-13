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
G = nx.karate_club_graph()
G = G.to_undirected()


temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)):
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]


num_init = 10
best_loss_raa = 1e16
best_loss_lsm = 1e16
seed_split = 1
seed_init = 42
iter = 1000 #500

lsm_accs_knn = []
lsm_accs_lg = []

for _ in range(num_init):
    lsm = LSM(d=2,
            sample_size=1,
            data = edge_list,
            data_type="edge list",
            link_pred=False)
    lsm.train(iterations=iter, LR=0.01, print_loss=False)
    #Saving best model
    if np.mean(lsm.losses[-100:]) < best_loss_lsm:
        best_lsm = lsm
        best_loss_lsm = np.mean(lsm.losses[-100:])
    lsm_knn_mu, lsm_knn_ci, lsm_knn_std = lsm.KNeighborsClassifier(list(nx.get_node_attributes(G, "club").values()))
    lsm_lg_mu, lsm_lg_ci, lsm_lg_std = lsm.logistic_regression(list(nx.get_node_attributes(G, "club").values()))
    lsm_accs_knn.append(lsm_knn_mu)
    lsm_accs_lg.append(lsm_lg_mu)

    seed_init += 1


RAA_avg_accs_knn = {}
RAA_conf_accs_knn = {}

RAA_avg_accs_lg = {}
RAA_conf_accs_lg = {}

kvals = [2,3,4,5,6,7,8]

for k in kvals:
    #seed_init = 1
    RAA_accs_knn = []
    RAA_accs_lg = []

    for _ in range(num_init):
        #define model
        kaa = KAA(k=k, 
              data=nx.adjacency_matrix(G).todense(),
              data_type="adjacency matrix"
              )
        kaa.train(iterations=100)
        RAA = DRRAA(d=2, 
                    k=k,
                    data=edge_list,
                    data_type='edge list',
                    link_pred=False,
                    sample_size=1,
                    init_Z=kaa.S.detach())

        RAA.train(iterations=iter, LR=0.01, print_loss=False, scheduling=False, early_stopping=0.8)
        #Saving best model
        if np.mean(RAA.losses[-100:]) < best_loss_raa:
            best_loss_raa = np.mean(RAA.losses[-100:])
            best_raa = RAA
        
        raa_knn_mu, raa_knn_ci, raa_knn_std = RAA.KNeighborsClassifier(list(nx.get_node_attributes(G, "club").values()))
        raa_lg_mu, raa_lg_ci, raa_lg_std = RAA.logistic_regression(list(nx.get_node_attributes(G, "club").values()))

        RAA_accs_knn.append(raa_knn_mu)
        RAA_accs_lg.append(raa_lg_mu)
        seed_init += 1
    

    RAA_avg_accs_knn[k] = np.mean(RAA_accs_knn)
    RAA_conf_accs_knn[k] = st.t.interval(alpha=0.95, df=len(RAA_accs_knn)-1, 
                    loc=np.mean(RAA_accs_knn), 
                    scale=st.sem(RAA_accs_knn))


    RAA_avg_accs_lg[k] = np.mean(RAA_accs_lg)
    RAA_conf_accs_lg[k] = st.t.interval(alpha=0.95, df=len(RAA_accs_lg)-1, 
                    loc=np.mean(RAA_accs_lg), 
                    scale=st.sem(RAA_accs_lg))


fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(kvals, list(RAA_avg_accs_knn.values()), '-o', label="RAA - KNN", color='C1')
ax.plot(kvals, [x for (x,y) in RAA_conf_accs_knn.values()], '--', color='C1')
ax.plot(kvals, [y for (x,y) in RAA_conf_accs_knn.values()], '--', color='C1')
ax.fill_between(kvals,
                 y1 = [x for (x,y) in RAA_conf_accs_knn.values()],
                 y2 = [y for (x,y) in RAA_conf_accs_knn.values()], color='C1', alpha=0.2)

ax.plot(kvals, list(RAA_avg_accs_lg.values()), '-o', label="RAA - LR", color='C2')
ax.plot(kvals, [x for (x,y) in RAA_conf_accs_lg.values()], '--', color='C2')
ax.plot(kvals, [y for (x,y) in RAA_conf_accs_lg.values()], '--', color='C2')
ax.fill_between(kvals,
                 y1 = [x for (x,y) in RAA_conf_accs_lg.values()],
                 y2 = [y for (x,y) in RAA_conf_accs_lg.values()], color='C2', alpha=0.2)

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

avg_lsm_accs_lg = [0]*len(kvals) + np.mean(lsm_accs_lg)
conf_lsm_accs_lg = st.t.interval(alpha=0.95, df=len(lsm_accs_lg)-1, 
                        loc=np.mean(lsm_accs_lg), 
                        scale=st.sem(lsm_accs_lg))

ax.plot(kvals, avg_lsm_accs_lg, '-o', label="LDM - LR", color="C4")
ax.fill_between(kvals,
                 y1 = [0]*len(kvals) + conf_lsm_accs_lg[0],
                 y2 = [0]*len(kvals) + conf_lsm_accs_lg[1],
                 color='C4', alpha=0.2)
ax.plot(kvals, [0]*len(kvals) + conf_lsm_accs_lg[0], '--', color='C4')
ax.plot(kvals,  [0]*len(kvals) + conf_lsm_accs_lg[1], '--', color='C4')
ax.legend()
ax.set_xlabel("k: Number of archetypes in models")
ax.set_ylabel("Accuracy")
plt.savefig("karate_accs_nodeclass.png")


#get colorlist
color_list = ["303638","f0c808","5d4b20","469374","9341b3","e3427d","e68653","ebe0b0","edfbba","ffadad","ffd6a5","fdffb6","caffbf","9bf6ff","a0c4ff","bdb2ff","ffc6ff","fffffc"]
color_list = ["#"+i.lower() for i in color_list]
color_map = [color_list[14] if G.nodes[i]['club'] == "Mr. Hi" else color_list[5] for i in G.nodes()]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10), dpi=100)
d = dict(G.degree)
raa_embeddings, raa_archetypes = best_raa.get_embeddings()
lsm_embeddings = best_lsm.get_embeddings()[0]
pos_map_raa = dict(list(zip(G.nodes(), list(raa_embeddings))))
pos_map_lsm = dict(list(zip(G.nodes(), list(lsm_embeddings))))
nx.draw_networkx_nodes(G, pos=pos_map_raa, ax = ax1, node_color=color_map, alpha=.9, node_size=[10*v for v in d.values()])
nx.draw_networkx_edges(G, pos=pos_map_raa, ax = ax1, alpha=.1)
nx.draw_networkx_nodes(G, pos=pos_map_lsm, ax=ax2, node_color=color_map, alpha=.9, node_size=[10*v for v in d.values()])
nx.draw_networkx_edges(G, pos=pos_map_lsm, ax=ax2, alpha=.1)
archetypal_nodes = best_raa.archetypal_nodes()
ax1.scatter(raa_archetypes[0, :], raa_archetypes[1, :], marker='^', c='black', label="Archetypes", s=80)
for i in archetypal_nodes:
    ax1.annotate(str(i), 
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
plt.savefig("karate_embeddings_nodeclass.png")

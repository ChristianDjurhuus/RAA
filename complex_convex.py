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
import scipy.stats as st
from tqdm import tqdm

seed = 1998
torch.random.manual_seed(seed)
np.random.seed(seed)

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return
setup_mpl()

iter = 10
avgNMIs = {}
avgAUCs = {}
avgIAUCs = {}
conf_NMIs = {}
conf_AUCs = {}
conf_IAUCs = {}
#Get synthetic data and convert to edge list
true_k = 8
true_alpha = 0.2
adj_m, z, A, Z_true = main(alpha=true_alpha, k=true_k, dim=2, nsamples=100) #z is cmap
G = nx.from_numpy_matrix(adj_m.numpy())
temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)): 
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

#edge_list = torch.FloatTensor(edge_list).long()
num_arc =  [2,3,4,5,6,7,8,9,10]
d = 2
for k in tqdm(num_arc):
    NMIs = []
    AUCs = []
    IAUCs = []
    for i in tqdm(range(10)):
        model = DRRAA(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    link_pred=True)

        model.train(iterations=iter, LR=0.01)
        auc_score, fpr, tpr = model.link_prediction()
        if k==8:
            ideal_score, _, _ = model.ideal_prediction(A, Z_true)
            IAUCs.append(ideal_score)
        AUCs.append(auc_score)
        Z = F.softmax(model.Z, dim=0)
        G = F.sigmoid(model.Gate)
        C = (Z.T * G) / (Z.T * G).sum(0)

        u, sigma, v = torch.svd(model.A) # Decomposition of A.
        r = torch.matmul(torch.diag(sigma), v.T)
        embeddings = torch.matmul(r, torch.matmul(torch.matmul(Z, C), Z)).T
        archetypes = torch.matmul(r, torch.matmul(Z, C))

        #if embeddings.shape[1] == 3:
        #    fig = plt.figure(dpi=100)
        #    ax = fig.add_subplot(projection='3d')
        #    sc = ax.scatter(embeddings[:, 0].detach().numpy(), embeddings[:, 1].detach().numpy(),
        #                embeddings[:, 2].detach().numpy(), c = z)
        #    ax.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(),
        #                archetypes[2, :].detach().numpy(), marker='^', c='black')
        #    fig.colorbar(sc, label="Density")
            #plt.show()
        #else:
        #    fig, ax = plt.subplots(dpi=100)
        #    sc = ax.scatter(embeddings[:, 0].detach().numpy(), embeddings[:, 1].detach().numpy(), c = z)
        #    ax.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(), marker='^', c='black')
        #    fig.colorbar(sc, label="Density")
            #plt.show()


        #Calculate NMI between embeddings
        NMIs.append(calcNMI(Z, Z_true).item())
        print(f'The NMI between z and z_hat is {calcNMI(Z, Z_true)}')
    avgNMIs[k] = np.mean(NMIs)
    conf_NMIs[k] =  st.t.interval(alpha=0.95, df=len(NMIs)-1, 
                    loc=np.mean(NMIs), 
                    scale=st.sem(NMIs)) 
    avgAUCs[k] = np.mean(AUCs)
    conf_AUCs[k] = st.t.interval(alpha=0.95, df=len(AUCs)-1, 
                    loc=np.mean(AUCs), 
                    scale=st.sem(AUCs))
    
    if k == true_k:
        avgIAUCs[k] = np.mean(IAUCs)
        conf_IAUCs[k] = st.t.interval(alpha=0.95, df=len(IAUCs)-1, 
                        loc=np.mean(IAUCs), 
                        scale=st.sem(IAUCs))

fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(num_arc, list(avgNMIs.values()), '-o', label="mean NMI with 95% CI")
ax.fill_between(num_arc,
                 y1 = [x for (x,y) in conf_NMIs.values()],
                 y2 = [y for (x,y) in conf_NMIs.values()],
                 color='tab:blue', alpha=0.2)

ax.axvline(true_k, linestyle = '--', color='r', label="True number of Archetypes", alpha=0.5)
ax.set_xlabel("k (Number of archetypes)")
ax.set_title(r"The NMI with different number of archetypes")
ax.set_ylabel("score")
ax.legend()
#plt.savefig("complex_NMI.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(num_arc, list(avgAUCs.values()), '-o', label="mean AUC with 95% CI")
ax.fill_between(num_arc,
                 y1 = [x for (x,y) in conf_AUCs.values()],
                 y2 = [y for (x,y) in conf_AUCs.values()],
                 color='tab:blue', alpha=0.2)
ax.axvline(8, linestyle = '--', color='r', label="True number of Archetypes", alpha=0.5)
ax.errorbar(8, list(avgIAUCs.values()), 
            [abs(x-y)/2 for (x,y) in conf_IAUCs.values()],
            solid_capstyle='projecting', capsize=5,
            label="mean ideal AUC with 95% CI", color='b')
ax.set_xlabel("k (Number of archetypes)")
ax.set_title("The AUC with different number of archetypes")
ax.set_ylabel("score")
ax.legend()
#plt.savefig("complex_AUC.pdf")
plt.show()


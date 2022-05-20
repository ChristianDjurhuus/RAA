'''
Random effects and gating (NMI): 
    Ideal predictor: True 
    Run with synthetic data K = 3 
    Alphas - 0.2, 1, 5. 
    N = 100 
    K = 3 
    D = 2 
    CV = 5 
    seed = 1998 
    Sample_size = 1 
    Lr = 0.01 (default)
    Iterations = 10,000 
'''
from src.data.synthetic_data import main
from src.models.calcNMI import calcNMI

from src.models.train_DRRAA_module import DRRAA
from src.models.train_DRRAA_ngating import DRRAA_ngating
from src.models.train_DRRAA_nre import DRRAA_nre
from src.models.train_DRRAA_bare import DRRAA_bare

from src.models.train_LSM_module import LSM
from src.models.train_LSM_module import LSMAA

from src.models.train_KAA_module import KAA

import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as st

##########################
## Synthetic data study ##
##########################


##############################################
# Investigation of Random effects and Gating #
#       Data without Random effects          #
##############################################
#TODO
# Increasing # archetypes
# Data with and without random effects (Degree heterogenity vs not)
seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

avgNMIs = {}
conf_NMIs = {}

avgNMIs_nr = {}
conf_NMIs_nr = {}

avgNMIs_ng = {}
conf_NMIs_ng = {}

avgNMIs_bare = {}
conf_NMIs_bare = {}

num_init = 5
k=3
d=2
nsamples=100
alphas = [0.2, 1, 5]
iter = 10000
for alpha in alphas:
    adj_m, z, A, Z_true, beta = main(alpha, k, d, nsamples, rand=False)
    Graph = nx.from_numpy_matrix(adj_m.numpy())
    temp = [x for x in nx.generate_edgelist(Graph, data=False)]
    edge_list = np.zeros((2, len(temp)))
    for i in range(len(temp)): 
        edge_list[0, i] = temp[i].split()[0]
        edge_list[1, i] = temp[i].split()[1]

    NMIs = []
    NMIs_ng = []
    NMIs_nr = []
    NMIs_bare = []

    for i in range(num_init):
        RAA = DRRAA(k=k,
                    d=d, 
                    sample_size=0.5, #Without random sampling
                    data=edge_list,
                    data_type='edge list')
        RAA_ng = DRRAA_ngating(k=k,
                    d=d, 
                    sample_size=0.5, #Without random sampling
                    data=edge_list,
                    data_type='edge list')
        
        RAA_nr = DRRAA_nre(k=k,
                    d=d, 
                    sample_size=0.5, #Without random sampling
                    data=edge_list,
                    data_type='edge list')
        
        RAA_bare = DRRAA_bare(k=k,
            d=d, 
            sample_size=0.5, #Without random sampling
            data=edge_list,
            data_type='edge list')

        RAA.train(iterations=iter, LR=0.01)
        RAA_ng.train(iterations=iter, LR=0.01)
        RAA_nr.train(iterations=iter, LR=0.01)
        RAA_bare.train(iterations=iter, LR=0.01)

        #RAA
        Z = F.softmax(RAA.Z, dim=0)
        #RAA no gating
        Z_ng = F.softmax(RAA_ng.Z, dim=0)
        #RAA no random effects
        Z_nr = F.softmax(RAA_nr.Z, dim=0)
        #RAA no gate no RE
        Z_bare = F.softmax(RAA_bare.Z, dim=0)

        NMIs.append(calcNMI(Z, Z_true).item())
        NMIs_ng.append(calcNMI(Z_ng, Z_true).item())
        NMIs_nr.append(calcNMI(Z_nr, Z_true).item())
        NMIs_bare.append(calcNMI(Z_bare, Z_true).item())

    #RAA
    avgNMIs[alpha] = np.mean(NMIs)
    conf_NMIs[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs)-1, 
                    loc=np.mean(NMIs), 
                    scale=st.sem(NMIs)) 

    #RAA no gating
    avgNMIs_ng[alpha] = np.mean(NMIs_ng)
    conf_NMIs_ng[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs_ng)-1, 
                    loc=np.mean(NMIs_ng), 
                    scale=st.sem(NMIs_ng)) 

    #RAA no random effects
    avgNMIs_nr[alpha] = np.mean(NMIs_nr)
    conf_NMIs_nr[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs_nr)-1, 
                    loc=np.mean(NMIs_nr), 
                    scale=st.sem(NMIs_nr)) 
    
    #RAA no gate and random effects
    avgNMIs_bare[alpha] = np.mean(NMIs_bare)
    conf_NMIs_bare[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs_bare)-1, 
                    loc=np.mean(NMIs_bare), 
                    scale=st.sem(NMIs_bare)) 

fig, ax = plt.subplots(figsize=(10,5), dpi=500)
ax.plot(alphas, list(avgNMIs.values()), '-o', label="RAA")
ax.plot(alphas, [x for (x,y) in conf_NMIs.values()], '--', color='C0')
ax.plot(alphas, [y for (x,y) in conf_NMIs.values()], '--', color='C0')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs.values()],
                 y2 = [y for (x,y) in conf_NMIs.values()], color='C0', alpha=0.2)

ax.plot(alphas, list(avgNMIs_ng.values()), '-o', label="RAA no gating")
ax.plot(alphas, [x for (x,y) in conf_NMIs_ng.values()], '--', color='C1')
ax.plot(alphas, [y for (x,y) in conf_NMIs_ng.values()], '--', color='C1')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs_ng.values()],
                 y2 = [y for (x,y) in conf_NMIs_ng.values()], color='C1', alpha=0.2)

ax.plot(alphas, list(avgNMIs_nr.values()), '-o', label="RAA no random effects")
ax.plot(alphas, [x for (x,y) in conf_NMIs_nr.values()], '--', color='C2')
ax.plot(alphas, [y for (x,y) in conf_NMIs_nr.values()], '--', color='C2')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs_nr.values()],
                 y2 = [y for (x,y) in conf_NMIs_nr.values()], color='C2', alpha=0.2)

ax.plot(alphas, list(avgNMIs_bare.values()), '-o', label="RAA no gating / random effects")
ax.plot(alphas, [x for (x,y) in conf_NMIs_bare.values()], '--', color='C3')
ax.plot(alphas, [y for (x,y) in conf_NMIs_bare.values()], '--', color='C3')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs_bare.values()],
                 y2 = [y for (x,y) in conf_NMIs_bare.values()], color='C3', alpha=0.2)                 

ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("NMI")
ax.grid(alpha=.3)
ax.legend()
plt.savefig("synthetic_test_nmi.png", dpi=500)
#plt.show()





##############################################
# Investigation of Random effects and Gating #
#       Data with Random effects             #
##############################################

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

avgAUCs = {}
avgNMIs = {}
conf_NMIs = {}
conf_AUCs = {}

avgAUCs_nr = {}
avgNMIs_nr = {}
conf_NMIs_nr = {}
conf_AUCs_nr = {}

avgAUCs_ng = {}
avgNMIs_ng = {}
conf_NMIs_ng = {}
conf_AUCs_ng = {}

avgAUCs_bare = {}
avgNMIs_bare = {}
conf_NMIs_bare = {}
conf_AUCs_bare = {}

for alpha in alphas:
    #Creating synth data
    adj_m, z, A, Z_true, beta = main(alpha, k, d, nsamples, rand=True)
    Graph = nx.from_numpy_matrix(adj_m.numpy())
    temp = [x for x in nx.generate_edgelist(Graph, data=False)]
    edge_list = np.zeros((2, len(temp)))
    for i in range(len(temp)): 
        edge_list[0, i] = temp[i].split()[0]
        edge_list[1, i] = temp[i].split()[1]
    

    NMIs = []
    NMIs_ng = []
    NMIs_nr = []
    NMIs_bare = []

    for i in range(num_init):
        RAA = DRRAA(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list')
        RAA_ng = DRRAA_ngating(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list')
        
        RAA_nr = DRRAA_nre(k=k,
                    d=d, 
                    sample_size=1, #Without random sampling
                    data=edge_list,
                    data_type='edge list')
        
        RAA_bare = DRRAA_bare(k=k,
            d=d, 
            sample_size=1, #Without random sampling
            data=edge_list,
            data_type='edge list')

        RAA.train(iterations=iter, LR=0.01)
        RAA_ng.train(iterations=iter, LR=0.01)
        RAA_nr.train(iterations=iter, LR=0.01)
        RAA_bare.train(iterations=iter, LR=0.01)

        #RAA
        Z = F.softmax(RAA.Z, dim=0)
        #RAA no gating
        Z_ng = F.softmax(RAA_ng.Z, dim=0)
        #RAA no random effects
        Z_nr = F.softmax(RAA_nr.Z, dim=0)
        #RAA no gate no RE
        Z_bare = F.softmax(RAA_bare.Z, dim=0)

        NMIs.append(calcNMI(Z, Z_true).item())
        NMIs_ng.append(calcNMI(Z_ng, Z_true).item())
        NMIs_nr.append(calcNMI(Z_nr, Z_true).item())
        NMIs_bare.append(calcNMI(Z_bare, Z_true).item())

    #RAA
    avgNMIs[alpha] = np.mean(NMIs)
    conf_NMIs[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs)-1, 
                    loc=np.mean(NMIs), 
                    scale=st.sem(NMIs)) 

    #RAA no gating
    avgNMIs_ng[alpha] = np.mean(NMIs_ng)
    conf_NMIs_ng[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs_ng)-1, 
                    loc=np.mean(NMIs_ng), 
                    scale=st.sem(NMIs_ng)) 

    #RAA no random effects
    avgNMIs_nr[alpha] = np.mean(NMIs_nr)
    conf_NMIs_nr[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs_nr)-1, 
                    loc=np.mean(NMIs_nr), 
                    scale=st.sem(NMIs_nr)) 
    
    #RAA no gate and random effects
    avgNMIs_bare[alpha] = np.mean(NMIs_bare)
    conf_NMIs_bare[alpha] =  st.t.interval(alpha=0.95, df=len(NMIs_bare)-1, 
                    loc=np.mean(NMIs_bare), 
                    scale=st.sem(NMIs_bare)) 


fig, ax = plt.subplots(figsize=(10,5), dpi=500)
ax.plot(alphas, list(avgNMIs.values()), '-o', label="RAA")
ax.plot(alphas, [x for (x,y) in conf_NMIs.values()], '--', color='C0')
ax.plot(alphas, [y for (x,y) in conf_NMIs.values()], '--', color='C0')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs.values()],
                 y2 = [y for (x,y) in conf_NMIs.values()], color='C0', alpha=0.2)

ax.plot(alphas, list(avgNMIs_ng.values()), '-o', label="RAA no gating")
ax.plot(alphas, [x for (x,y) in conf_NMIs_ng.values()], '--', color='C1')
ax.plot(alphas, [y for (x,y) in conf_NMIs_ng.values()], '--', color='C1')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs_ng.values()],
                 y2 = [y for (x,y) in conf_NMIs_ng.values()], color='C1', alpha=0.2)

ax.plot(alphas, list(avgNMIs_nr.values()), '-o', label="RAA no random effects")
ax.plot(alphas, [x for (x,y) in conf_NMIs_nr.values()], '--', color='C2')
ax.plot(alphas, [y for (x,y) in conf_NMIs_nr.values()], '--', color='C2')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs_nr.values()],
                 y2 = [y for (x,y) in conf_NMIs_nr.values()], color='C2', alpha=0.2)

ax.plot(alphas, list(avgNMIs_bare.values()), '-o', label="RAA no gating / random effects")
ax.plot(alphas, [x for (x,y) in conf_NMIs_bare.values()], '--', color='C3')
ax.plot(alphas, [y for (x,y) in conf_NMIs_bare.values()], '--', color='C3')
ax.fill_between(alphas,
                 y1 = [x for (x,y) in conf_NMIs_bare.values()],
                 y2 = [y for (x,y) in conf_NMIs_bare.values()], color='C3', alpha=0.2)                 

ax.set_xlabel(r"$\alpha$: Parameter of the Dirichlet Distribution")
ax.set_ylabel("NMI")
ax.grid(alpha=.3)
ax.legend()
plt.savefig("synthetic_test_nmi_re.png", dpi=500)
#plt.show()
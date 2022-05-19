'''
LDM with AA on embeddings vs. RAA:
		Run with synthetic data K = 3
		Synthetic alphas = 0.2
		N = 100 (wip)
		K = 2 .. 10
		D = 2
		Inits = 5           #Number of inits.
		seed = 1998
		sample_size = 1
		Lr = 0.01
		Iterations = 10,000
'''
from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSMAA
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl 
from src.data.synthetic_data import main
from src.data.synthetic_data import ideal_prediction
import networkx as nx

seed = 1998
torch.random.manual_seed(seed)
np.random.seed(seed)

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return
setup_mpl()

#################################
## Synthetic model comparison  ##
##      RAA and baselines      ##
#################################

#Data creation
real_alpha = 0.2
K = 3
n = 100
d = 2
adj_m, z, A, Z_true, beta = main(alpha=real_alpha, k=K, dim=d, nsamples=n, rand=False)
G = nx.from_numpy_matrix(adj_m.numpy())

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)): 
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

#Defining models
iter = 10000
num_init = 5
kvals = [2,3,4,5,6,7,8]

avg_raa_aucs = {}
avg_ldmaa_aucs = {}
conf_raa_aucs = {}
conf_ldmaa_aucs = {}

Iaucs = []


for _ in range(num_init):
    #Prediction with ideal embeddings
    ideal_score, _, _ = ideal_prediction(adj_m, A, Z_true, beta=beta, test_size = 0.3)
    Iaucs.append(ideal_score)

for kval in kvals:
    raa_aucs = []
    ldmaa_aucs = []
    for _ in range(num_init):
        raa = DRRAA(k=kval,
                    d=d,
                    sample_size=1,
                    data=edge_list,
                    data_type = "edge list",
                    link_pred = True
        )
        raa.train(iterations=iter)
        raa_auc, _, _ = raa.link_prediction()
        raa_aucs.append(raa_auc)

        ldmaa = LSMAA(d=d,
                      k=kval,
                      sample_size=1,
                      data = edge_list,
                      data_type = "Edge list",
                      link_pred = True)
        ldmaa.train(iterations=iter)
        ldmaa_auc, _, _ = ldmaa.link_prediction()
        ldmaa_aucs.append(ldmaa_auc)

    avg_raa_aucs[kval] = np.mean(raa_aucs)
    avg_ldmaa_aucs[kval] = np.mean(ldmaa_aucs)
    conf_raa_aucs[kval] = st.t.interval(alpha=0.95, df=len(raa_aucs)-1, 
                        loc=np.mean(raa_aucs), 
                        scale=st.sem(raa_aucs))
    conf_ldmaa_aucs[kval] = st.t.interval(alpha=0.95, df=len(ldmaa_aucs)-1, 
                        loc=np.mean(ldmaa_aucs), 
                        scale=st.sem(ldmaa_aucs))


fig, ax = plt.subplots(figsize=(10,5), dpi=500)
ax.plot(kvals, avg_raa_aucs.values(), '-o', label="RAA", color='C1')
ax.fill_between(kvals,
                 y1 = [x for (x,y) in conf_raa_aucs.values()],
                 y2 = [y for (x,y) in conf_raa_aucs.values()],
                 color='C1', alpha=0.2)
ax.plot(kvals, [x for (x,y) in conf_raa_aucs.values()], '--', color='C1')
ax.plot(kvals, [y for (x,y) in conf_raa_aucs.values()], '--', color='C1')

ax.plot(kvals, avg_ldmaa_aucs.values(), '-o', label="ldmaa (Jaccard)", color='C2')
ax.fill_between(kvals,
                 y1 = [x for (x,y) in conf_ldmaa_aucs.values()],
                 y2 = [y for (x,y) in conf_ldmaa_aucs.values()],
                 color='C2', alpha=0.2)
ax.plot(kvals, [x for (x,y) in conf_ldmaa_aucs.values()], '--', color='C2')
ax.plot(kvals, [y for (x,y) in conf_ldmaa_aucs.values()], '--', color='C2')



conf_Iaucs = st.t.interval(alpha=0.95, df=len(Iaucs)-1, 
                        loc=np.mean(Iaucs), 
                        scale=st.sem(Iaucs))
ax.plot(K,np.mean(Iaucs),'bo', markersize=5)
ax.errorbar(K, np.mean(Iaucs), 
            [abs(x-y)/2 for (x,y) in [conf_Iaucs]],
            solid_capstyle='projecting', capsize=5,
            label="Ideal AUC", color='b')

ax.axvline(K, linestyle = '--', color='C4', label="True number of Archetypes", alpha=0.5)
ax.grid(alpha=.3)
ax.set_xlabel("k: Number of archetypes")
ax.set_ylabel("AUC")
ax.legend()
plt.savefig('two_step_vs_one_step.png',dpi=500)
plt.show()

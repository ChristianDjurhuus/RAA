from src.models.train_DRRAA_module import DRRAA
from src.models.train_DRRAA_ngating import DRRAA_ngating
from src.models.train_DRRAA_nre import DRRAA_nre
from src.models.train_DRRAA_bare import DRRAA_bare
from src.models.calcNMI import calcNMI
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl
from src.data.synthetic_data import main
from src.data.synthetic_data import ideal_prediction
import networkx as nx
import archetypes as arch
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

rand = True
if rand:
    np.random.seed(1)
    torch.manual_seed(1)
else:
    np.random.seed(42)
    torch.manual_seed(42)


real_alpha = 0.1
K = 3
n = 1000
d = 2
adj_m, z, A, Z_true, beta, partition = main(alpha=real_alpha, k=K, dim=d, nsamples=n, rand=rand)
G = nx.from_numpy_matrix(adj_m.numpy())

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)):
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

#set test and train split seed. We want the same train and test split in order to know that differences
#are because of inits.
seed_split = 42

iter = 5000
#Z_s=Z_true
raa = DRRAA(k=K,
                    d=d,
                    sample_size=0.4,
                    data=edge_list,
                    data_type = "edge list",
                    link_pred = False,
                    seed_init=60,
        )
raa.train(iterations=iter, print_loss=True, LR=0.1)

print(calcNMI(F.softmax(raa.Z.detach(),dim=0), Z_true).float().item())



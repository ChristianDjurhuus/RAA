from src.models.train_KAA_module import KAA
from src.models.train_DRRAA_module import DRRAA
from src.models.synthetic_data import main
import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt

k = 2
d = 2
alpha = 0.2
iter = 10
#adj_m, z, A, Z_true = main(alpha, k=k, N=1000) #z is cmap
#G = nx.from_numpy_matrix(adj_m.numpy())
#G = nx.karate_club_graph()
data = 'data/raw/polblogs/polblogs.gml'
node_attribute = 'value'

#temp = [x for x in nx.generate_edgelist(G, data=False)]
#edge_list = np.zeros((2, len(temp)))
#for i in range(len(temp)): 
#    edge_list[0, i] = temp[i].split()[0]
#    edge_list[1, i] = temp[i].split()[1]

#edge_list = torch.from_numpy(edge_list).long()

#model = KAA(k=k, data=adj_m, type='jaccard')
#model.train(iterations=iter)
#model.link_prediction()

model = DRRAA(k=k,
                d=d, 
                sample_size=1, #Without random sampling
                data_type='gml',
                data = data,
                test_size=0.2)
model.train(iterations=iter)
auc_score, fpr, tpr = model.link_prediction()
print(auc_score)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
plt.plot([0, 1], [0, 1],'r--', label='random')
plt.legend(loc = 'lower right')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("RAA model")
plt.show()




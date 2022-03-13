from collections import defaultdict
import networkx as nx
import torch
from numpy import zeros
# converts from adjacency matrix to adjacency list
seed = 1984
torch.random.manual_seed(seed)

ZKC_graph = nx.karate_club_graph()
X = nx.convert_matrix.to_numpy_matrix(ZKC_graph)
X = torch.from_numpy(X)

def convert(a):
    adjList = defaultdict(list)
    for i in range(len(a)):
        for j in range(len(a[i])):
                       if a[i][j]== 1:
                           adjList[i].append(j)
    return adjList

edge_list = convert(X)


X_shape = X.shape
num_samples = 15
idx_i_test = torch.multinomial(input=torch.arange(0, float(X_shape[0])), num_samples=num_samples,
                                replacement=True)
idx_j_test = torch.tensor(zeros(num_samples)).long()
for i in range(len(idx_i_test)):
    idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(X_shape[1]))[
        torch.multinomial(input=torch.arange(idx_i_test[i].item(), float(X_shape[1])), num_samples=1,
                            replacement=True).item()].item()  # Temp solution to sample from upper corner

target = torch.tensor(zeros(num_samples)).long()
for i in range(num_samples):
    if idx_j_test[i].item() in edge_list[idx_i_test[i].item()]:
        target[i] = 1

print(target)



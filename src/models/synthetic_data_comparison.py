from synthetic_data import main
from synthetic_data import synthetic_data
from train_DRRAA import DRRAA
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

seed = 4
torch.random.manual_seed(seed)
#Get synthetic data and convert to edge list
adj_m, z = main() #z is cmap
G = nx.from_numpy_matrix(adj_m.numpy())

temp = [x for x in nx.generate_edgelist(G, data=False)]
edge_list = np.zeros((2, len(temp)))
for i in range(len(temp)):
    edge_list[0, i] = temp[i].split()[0]
    edge_list[1, i] = temp[i].split()[1]

edge_list = torch.FloatTensor(edge_list).long()
N = 100
k = 3
d = 3

#Train model
model = DRRAA(input_size = (N, N),
                k=k,
                d=d, 
                sampling_weights=torch.ones(N), 
                sample_size=round(N), #Without random sampling
                edge_list=edge_list)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

losses = []
iterations = 1000
for _ in range(iterations):
    loss = - model.log_likelihood() / model.input_size[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print('Loss at the',_,'iteration:',loss.item())

Z = F.softmax(model.Z, dim=0)
G = F.sigmoid(model.G)
C = (Z.T * G) / (Z.T * G).sum(0)

embeddings = torch.matmul(model.A, torch.matmul(torch.matmul(Z, C), Z)).T
archetypes = torch.matmul(model.A, torch.matmul(Z, C))

if embeddings.shape[1] == 3:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(embeddings[:, 0].detach().numpy(), embeddings[:, 1].detach().numpy(),
                embeddings[:, 2].detach().numpy())
    ax.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(),
                archetypes[2, :].detach().numpy(), marker='^', c='black')
    fig.colorbar(sc, label="Density")
plt.show()



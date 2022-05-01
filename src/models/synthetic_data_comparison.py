from synthetic_data import main
from synthetic_data import synthetic_data
from train_DRRAA import DRRAA
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from calcNMI import calcNMI
import matplotlib as mpl

AUCs = []
NMIs = []
seed = 42
torch.random.manual_seed(seed)
alpha_values = np.linspace(0.1, 1.7, 8)
for idx, alpha in enumerate(alpha_values):
    print(f'Iteration number {idx} with alpha {alpha}')
    #Get synthetic data and convert to edge list
    adj_m, z, A, Z_true = main(alpha) #z is cmap
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

    link_pred = True

    if link_pred:
        num_samples = round(0.2*N)
        idx_i_test = torch.multinomial(input=torch.arange(0, float(N)), num_samples=num_samples,
                                        replacement=True)
        idx_j_test = torch.tensor(np.zeros(num_samples)).long()
        for i in range(len(idx_i_test)):
            idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(N))[
                torch.multinomial(input=torch.arange(idx_i_test[i].item(), float(N)), num_samples=1,
                                    replacement=True).item()].item()  # Temp solution to sample from upper corner

        test = torch.stack((idx_i_test,idx_j_test))

        #TODO: could be a killer.. maybe do it once and save adjacency list ;)
        def if_edge(a, edge_list):
            a = a.tolist()
            edge_list = edge_list.tolist()
            a = list(zip(a[0], a[1]))
            edge_list = list(zip(edge_list[0], edge_list[1]))
            return [a[i] in edge_list for i in range(len(a))]

        target = if_edge(test, edge_list)

    #Train model
    model = DRRAA(input_size = (N, N),
                    k=k,
                    d=d, 
                    sampling_weights=torch.ones(N), 
                    sample_size=round(N), #Without random sampling
                    edge_list=edge_list)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    losses = []
    iterations = 5000
    for _ in range(iterations):
        loss = - model.log_likelihood() / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Loss at the',_,'iteration:',loss.item())

    #Link prediction
    if link_pred:
        auc_score, fpr, tpr = model.link_prediction(target, idx_i_test, idx_j_test)
        #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
        #plt.plot([0, 1], [0, 1],'r--', label='random')
        #plt.legend(loc = 'lower right')
        #plt.xlabel("False positive rate")
        #plt.ylabel("True positive rate")
        #plt.title("RAA model")
        #plt.show()
        AUCs.append(auc_score)

    Z = F.softmax(model.Z, dim=0)
    G = F.sigmoid(model.G)
    C = (Z.T * G) / (Z.T * G).sum(0)


    embeddings = torch.matmul(model.A, torch.matmul(torch.matmul(Z, C), Z)).T
    archetypes = torch.matmul(model.A, torch.matmul(Z, C))

    #if embeddings.shape[1] == 3:
    #    fig = plt.figure()
    #    ax = fig.add_subplot(projection='3d')
    #    sc = ax.scatter(embeddings[:, 0].detach().numpy(), embeddings[:, 1].detach().numpy(),
    #                embeddings[:, 2].detach().numpy(), c = z)
    #    ax.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(),
    #                archetypes[2, :].detach().numpy(), marker='^', c='black')
    #    fig.colorbar(sc, label="Density")
    #plt.show()

    #Calculate NMI between embeddings
    NMIs.append(calcNMI(Z, Z_true).item())
    #print(f'The NMI between z and z_hat is {calcNMI(Z, Z_true)}')

mpl.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(alpha_values, NMIs, label='NMIs')
ax.set_xlabel("alpha value")
ax.set_title("The NMI with different alpha values")
ax.set_ylabel("score")
ax.legend()
plt.show()
fig, ax = plt.subplots(figsize=(10,5), dpi=100)
ax.plot(alpha_values, AUCs, label='AUCs')
ax.set_xlabel("alpha value")
ax.set_title("The AUC with different alpha values")
ax.set_ylabel("score")
ax.legend()
plt.show()

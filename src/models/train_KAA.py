import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import networkx as nx




#Inspired from Mørup's implementation http://www.mortenmorup.dk/MMhomepageUpdated_files/Page327.htm

class KAA(nn.Module):
    def __init__(self, K, k, I, U):
        super(KAA, self).__init__()
        self.K = K.float()
        self.k = k
        self.I = I
        self.U = U
        self.S = torch.nn.Parameter(torch.randn(self.k, len(self.U)))
        self.C = torch.nn.Parameter(torch.randn(len(self.I), self.k))
        self.SST = torch.sum(self.K[:, self.U] * self.K[:, self.U])


    def log_likelihood(self,):
        S = F.softmax(self.S, dim=0)
        C = F.softmax(self.C, dim=0)
        KC = torch.matmul(self.K, C) # N x k
        CtKC = torch.matmul(torch.matmul(C.T, self.K), C) # k x k Might be a mistake here
        SSt = torch.matmul(S, S.T)
        SSE = self.SST - 2 * torch.sum(torch.sum(S.T * KC) + torch.sum(torch.sum(CtKC * SSt)))
        return SSE


if __name__ == "__main__":
    seed = 1984
    torch.random.manual_seed(seed)

    #A = mmread("data/raw/soc-karate.mtx")
    #A = A.todense()
    ZKC_graph = nx.karate_club_graph()
    #Let's keep track of which nodes represent John A and Mr Hi
    Mr_Hi = 0
    John_A = 33

    #Let's display the labels of which club each member ended up joining
    club_labels = nx.get_node_attributes(ZKC_graph,'club')

    #Getting adjacency matrix
    A = nx.convert_matrix.to_numpy_matrix(ZKC_graph)
    A = torch.from_numpy(A)

    k = 2 # number of archetypes
    K = torch.matmul(A.T, A)

    input_size = K.shape

    
    model = KAA(K = K, k=k, I=range(input_size[1]), U=range(input_size[1]))
    optimizer = torch.optim.Adam(params=model.parameters())
    
    losses = []
    iterations = 10000
    for _ in range(iterations):
        loss = - model.log_likelihood() / input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Loss at the',_,'iteration:',loss.item())
    


    embeddings = torch.matmul(A.float(), F.softmax(model.C, dim=0)).detach().numpy().T
    archetypes = torch.matmul(F.softmax(model.S, dim=0), F.softmax(model.C, dim=0)).detach().numpy()

    labels = list(club_labels.values())
    idx_hi = [i for i, x in enumerate(labels) if x == "Mr. Hi"]
    idx_of = [i for i, x in enumerate(labels) if x == "Officer"]

    if embeddings.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(embeddings[:,0].detach().numpy()[idx_hi], embeddings[:,1].detach().numpy()[idx_hi], embeddings[:,2].detach().numpy()[idx_hi], c = 'red', label='Mr. Hi' )
        ax.scatter(embeddings[:,0].detach().numpy()[idx_of], embeddings[:,1].detach().numpy()[idx_of], embeddings[:,2][idx_of].detach().numpy(), c = 'blue', label='Officer')
        ax.scatter(archetypes[0,:].detach().numpy(), archetypes[1,:].detach().numpy(), archetypes[2,:].detach().numpy(), marker = '^', c='black')
        ax.text(embeddings[Mr_Hi,0].detach().numpy(), embeddings[Mr_Hi,1].detach().numpy(), embeddings[Mr_Hi,2].detach().numpy(), 'Mr. Hi')
        ax.text(embeddings[John_A, 0].detach().numpy(), embeddings[John_A, 1].detach().numpy(), embeddings[John_A, 2].detach().numpy(),  'Officer')
        ax.set_title(f"Latent space after {iterations} iterations")
        ax.legend()
    else:
        fig, ax1 = plt.subplots()
        ax1.scatter(embeddings[0,:][idx_hi], embeddings[1,:][idx_hi], c = 'red', label='Mr. Hi')
        ax1.scatter(embeddings[0,:][idx_of], embeddings[1,:][idx_of], c = 'blue', label='Officer')
        ax1.scatter(archetypes[0,:], archetypes[0,:], marker = '^', c = 'black')
        ax1.annotate('Mr. Hi', embeddings[:,Mr_Hi])
        ax1.annotate('Officer', embeddings[:,John_A])
        ax1.legend()
        ax1.set_title("KAA")
    plt.show()
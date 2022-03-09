import torch
import torch.nn as nn
from scipy.io import mmread
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import networkx as nx 
import numpy as np
import umap
import umap.plot

class LSM(nn.Module):
    def __init__(self, A, input_size, latent_dim):
        super(LSM, self).__init__()
        self.A = A
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.alpha = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.latent_Z = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim))


    def random_sampling(self):
        #TODO

        return None

    def log_likelihood(self):

        z_dist = ((self.latent_Z.unsqueeze(1) - self.latent_Z + 1e-06)**2).sum(-1)**0.5 # (N x N)
        theta = self.alpha - z_dist #(N x N)
        softplus_theta = F.softplus(theta) # log(1+exp(theta))
        LL = 0.5 * (theta * self.A).sum() - 0.5 * torch.sum(softplus_theta-torch.diag(torch.diagonal(softplus_theta))) #Times by 0.5 to avoid double counting

        return LL
    
    def link_prediction(self, A_test, idx_i_test, idx_j_test):
        with torch.no_grad():

            z_pdist_test = ((self.latent_Z[idx_i_test, :].unsqueeze(1) - self.latent_Z[idx_j_test, :] + 1e-06)**2).sum(-1)**0.5 # N x N 
            theta = self.alpha[idx_i_test] + self.alpha[idx_j_test] - z_pdist_test #(N x N)

            #Get the rate -> exp(log_odds) 
            rate = torch.exp(theta).flatten() # N^2 

            #Create target (make sure its in the right order by indexing)
            target = A_test[idx_i_test.unsqueeze(1), idx_j_test].flatten() #N^2

            fpr, tpr, threshold = metrics.roc_curve(target.numpy(), rate.numpy())

            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target.cpu().data.numpy(), rate.cpu().data.numpy())

            return auc_score, fpr, tpr


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
    latent_dim = 2

    link_pred = True

    if link_pred:
        A_shape = A.shape
        num_samples = 10
        idx_i_test = torch.multinomial(input=torch.arange(0,float(A_shape[0])), num_samples=num_samples,
                                       replacement=True)
        idx_j_test = torch.multinomial(input=torch.arange(0, float(A_shape[1])), num_samples=num_samples,
                                       replacement=True)
        A_test = A.detach().clone()
        A_test[:] = 0
        A_test[idx_i_test, idx_j_test] = A[idx_i_test,idx_j_test]
        A[idx_i_test, idx_j_test] = 0

    model = LSM(A = A, input_size = A.shape, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(params=model.parameters())
    
    losses = []
    iterations = 10000
    for _ in range(iterations):
        loss = - model.log_likelihood() / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Loss at the',_,'iteration:',loss.item())
    
    #Link prediction
    if link_pred:
        auc_score, fpr, tpr = model.link_prediction(A_test, idx_i_test, idx_j_test)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
        plt.plot([0, 1], [0, 1],'r--', label='random')
        plt.legend(loc = 'lower right')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Latent space model")
        plt.show()


    labels = list(club_labels.values())
    idx_hi = [i for i, x in enumerate(labels) if x == "Mr. Hi"]
    idx_of = [i for i, x in enumerate(labels) if x == "Officer"]

    latent_Z = model.latent_Z.detach().numpy()

    if latent_Z.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(latent_Z[:,0][idx_hi], latent_Z[:,1][idx_hi], latent_Z[:,2][idx_hi], c = 'red', label='Mr. Hi' )
        ax.scatter(latent_Z[:,0][idx_of], latent_Z[:,1][idx_of], latent_Z[:,2][idx_of], c = 'blue', label='Officer')
        ax.text(latent_Z[Mr_Hi,0], latent_Z[Mr_Hi,1], latent_Z[Mr_Hi,2], 'Mr. Hi')
        ax.text(latent_Z[John_A, 0], latent_Z[John_A, 1], latent_Z[John_A, 2],  'Officer')
        ax.set_title(f"Latent space after {iterations} iterations")
        ax.legend()
        plt.show()

    if latent_Z.shape[1] == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.scatter(latent_Z[:,0][idx_hi], latent_Z[:,1][idx_hi], c = 'red', label='Mr. Hi')
        ax1.scatter(latent_Z[:,0][idx_of], latent_Z[:,1][idx_of], c = 'blue', label='Officer')
        ax1.annotate('Mr. Hi', latent_Z[Mr_Hi,:])
        ax1.annotate('Officer', latent_Z[John_A, :])
        ax1.legend()
        ax1.set_title(f"Latent space after {iterations} iterations")
        #Plotting learning curve
        ax2.plot(losses)
        ax2.set_title("Loss")
        plt.show()
        #Trying to add networkx drawing
        #pos = {i: latent_Z[i, :] for i in range(A.shape[0])}
        #nx.draw(ZKC_graph, with_labels=True, pos=pos)
        #plt.show()

    if latent_Z.shape[1] > 3:
        embedding = umap.UMAP().fit_transform(latent_Z)
        color_dict = {"Mr. Hi":"red", "Officer":"blue"}
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[color_dict[i] for i in labels ]
            )
        plt.annotate('Mr. Hi', embedding[Mr_Hi,:])
        plt.annotate('Officer', embedding[John_A, :])
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'UMAP projection of the latent space with dim: {latent_Z.shape[1]}')
        plt.show()
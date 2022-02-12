from re import I
from sklearn.utils import compute_sample_weight
import torch
import torch.nn as nn
from scipy.io import mmread
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

class RAA(nn.Module):
    def __init__(self, A, input_size, k):
        super(RAA, self).__init__()
        self.A = A
        self.input_size = input_size
        self.k = k

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.a = torch.nn.Parameter(torch.randn(1))
        #self.Z = torch.nn.Parameter(F.softmax(torch.randn(self.k, self.input_size[0]), dim=1)) #This maybe gives a better initialization but not sure if worth the compute
        #self.C = torch.nn.Parameter(F.softmax(torch.randn(self.input_size[0], self.k), dim=1)) 
        self.Z = torch.nn.Parameter(torch.randn(self.k, self.input_size[0]))
        self.C = torch.nn.Parameter(torch.randn(self.input_size[0], self.k))

    def random_sampling(self):
        #TODO

        return None

    def log_likelihood(self):

        beta = self.beta.unsqueeze(1) + self.beta #(N x N)
        Z = F.softmax(self.Z, dim=0) #(K x N)
        C = F.softmax(self.C, dim=0) #(N x K)
        M = torch.matmul(torch.matmul(Z, C), Z).T #(N x K)
        z_dist = ((M.unsqueeze(1) - M + 1e-06)**2).sum(-1)**0.5 # (N x N)
        theta = beta - self.a * z_dist #(N x N)
        softplus_theta = F.softplus(theta) # log(1+exp(theta))
        LL = ((theta-torch.diag(torch.diagonal(theta))) * self.A).sum() - torch.sum(softplus_theta-torch.diag(torch.diagonal(softplus_theta)))

        return LL

    def forward(self):


        return
    
    def link_prediction(self, A_test, idx_i_test, idx_j_test):
        with torch.no_grad():
            Z = F.softmax(self.Z, dim=0)
            C = F.softmax(self.C, dim=0)

            M_i = torch.matmul(torch.matmul(Z, C), self.Z[:, idx_i_test]).T #Size of test set e.g. K x N
            M_j = torch.matmul(torch.matmul(Z, C), self.Z[:, idx_j_test]).T
            z_pdist_test = ((M_i.unsqueeze(1) - M_j + 1e-06)**2).sum(-1)**0.5 # N x N 
            theta = (self.beta[idx_i_test] + self.beta[idx_j_test] - self.a * z_pdist_test) # N x N

            #Get the rate -> exp(log_odds) 
            rate = torch.exp(theta).flatten() # N^2 

            #Create target (make sure its in the right order by indexing)
            target = A_test[idx_i_test.unsqueeze(1), idx_j_test].flatten() #N^2


            fpr, tpr, threshold = metrics.roc_curve(target.numpy(), rate.numpy())


            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target.cpu().data.numpy(), rate.cpu().data.numpy())

            return auc_score, fpr, tpr


if __name__ == "__main__": 
    seed = 1998
    torch.random.manual_seed(seed)

    A = mmread("data/raw/soc-karate.mtx")
    A = A.todense()
    A = torch.from_numpy(A)
    k = 2

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

    model = RAA(A = A, input_size = A.shape, k=k)
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
    base_fpr = np.linspace(0, 1, 101)
    if link_pred:
        auc_score, fpr, tpr = model.link_prediction(A_test, idx_i_test, idx_j_test)
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        #tpr = np.interp(base_fpr, fpr, tpr)

    #Plotting latent space
    Z = F.softmax(model.Z, dim=0)
    C = F.softmax(model.C, dim=0)
    embeddings = torch.matmul(torch.matmul(Z, C), Z).T
    archetypes = torch.matmul(Z, C)

    if embeddings.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(embeddings[:,0].detach().numpy(), embeddings[:,1].detach().numpy(), embeddings[:,2].detach().numpy())
        ax.scatter(archetypes[:,0].detach().numpy(), archetypes[:,1].detach().numpy(), archetypes[:,2].detach().numpy(), marker = '^', c='red')
        ax.set_title(f"Latent space after {iterations} iterations")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=400)
        ax1.scatter(embeddings[:,0].detach().numpy(), embeddings[:,1].detach().numpy())
        #ax1.scatter(archetypes[:,0].detach().numpy(), archetypes[:,1].detach().numpy(), marker = '^', c = 'red')
        ax1.scatter(archetypes[0,:].detach().numpy(), archetypes[1,:].detach().numpy(), marker = '^', c = 'red')
        ax1.set_title(f"Latent space after {iterations} iterations")
        #Plotting learning curve
        ax2.plot(losses)
        ax2.set_title("Loss")
    plt.show()


















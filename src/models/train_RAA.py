import torch
import torch.nn as nn
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import Softmax
import torch.nn.functional as F

class RAA(nn.Module):
    def __init__(self, A, input_size, k):
        super(RAA, self).__init__()
        self.A = A
        self.input_size = input_size
        self.k = k

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.a = torch.nn.Parameter(torch.randn(1))
        self.Z = torch.nn.Parameter(torch.randn(self.k, self.input_size[0]))
        self.C = torch.nn.Parameter(torch.randn(self.input_size[0], self.k))

    def log_likelihood(self):

        beta = self.beta.unsqueeze(1) + self.beta #(N x N)
        Z = F.softmax(self.Z, dim=0) #(K x N)
        M = torch.matmul(torch.matmul(Z, F.softmax(self.C, dim=0)), Z).T #(N x K)
        z_dist = ((M.unsqueeze(1) - M + 1e-06)**2).sum(-1)**0.5 # (N x N)
        theta = beta - self.a * z_dist #(N x N)
        softplus_theta = F.softplus(theta) # log(1+exp(theta))
        LL = ((theta-torch.diag(torch.diagonal(theta))) * self.A).sum() - torch.sum(softplus_theta-torch.diag(torch.diagonal(softplus_theta)))

        return LL

if __name__ == "__main__": 
    seed = 1998
    torch.random.manual_seed(seed)

    A = mmread("data/raw/soc-karate.mtx")
    A = A.todense()
    A = torch.from_numpy(A)
    k = 2

    model = RAA(A = A, input_size = A.shape, k=k)
    optimizer = torch.optim.Adam(params=model.parameters())
    
    losses = []
    iterations = 50000
    for _ in range(iterations):
        loss = - model.log_likelihood() / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Loss at the',_,'iteration:',loss.item())
    
    def setup_mpl():
        mpl.rcParams['font.family'] = 'Helvetica Neue'
    setup_mpl()

    #Plotting latent space
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=400)
    #latent_zi = model.latent_zi.data.numpy()
    #latent_zj = model.latent_zj.data.numpy()
    #ax1.scatter(latent_zi[:,0], latent_zj[:,1])
    #ax1.set_title(f"Latent space after {iterations} iterations")
    #Plotting learning curve
    ax2.plot(losses)
    ax2.set_title("Loss")
    plt.show()


















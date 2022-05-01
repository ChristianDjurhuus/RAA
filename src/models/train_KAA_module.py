import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.metrics import jaccard_score
from torch_sparse import spspmm
import numpy as np

# import modules
from src.visualization.visualize import Visualization
from src.features.link_prediction import Link_prediction
from src.features.preprocessing import Preprocessing


class KAA(nn.Module):
    def __init__(self, k, data, type = "jaccard", data_type = "Adjacency matrix", data_2 = None, link_pred = False, ):
        super(KAA, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"
        Preprocessing.__init__(self, data = data, data_type = data_type, device = self.device, data_2 = data_2)
        self.edge_list, self.N = Preprocessing.convert_to_egde_list(self)
        #Link_prediction.__init__(self)
        #Visualization.__init__(self)

        self.X = data
        self.N = self.X.shape[0]
        self.input_size = (self.N, self.N)
        self.k = k
        self.type = type
        self.K = self.kernel(self.X, type = self.type)
        self.S = torch.nn.Parameter(torch.randn(self.k, self.input_size[0], device = self.device))
        self.C = torch.nn.Parameter(torch.randn(self.input_size[0], self.k, device = self.device))
        self.a = torch.nn.Parameter(torch.randn(1, device = self.device))
        self.losses = []

  
    def kernel(self, X, type):
        # check pairwise_distances
        #kernel = X.T@X
        if type == 'jaccard':
            kernel = 1-torch.from_numpy(pairwise_distances(X.T, X, metric=type)).float()
        if type == 'parcellating': #TODO: Does not seem to learn the structure.
            temp = ((X.unsqueeze(1) - X + 1e-06)**2).sum(-1)
            kernel = (2 * (temp - torch.diag(torch.diagonal(temp))))**0.5
        if type == 'normalised_x':
            kernel = X @ X.T / (X @ X.T).sum(0) #TODO: Sum row or column wise?
        if type == 'laplacian':
            D = torch.diag(X.sum(1))
            kernel = D - X #TODO: weird space..
        return kernel

    def SSE(self):
        S = torch.softmax(self.S, dim=0)
        C = torch.softmax(self.C, dim=0)
        KC = self.K @ C 
        CtKC = C.T @ self.K @ C
        SSt = S @ S.T
        SSE = - 2 * torch.sum( torch.sum( S.T *  KC)) + torch.sum(torch.sum(CtKC * SSt))
        return SSE

    def train(self, iterations, LR = 0.01, print_loss = True):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=LR)

        for _ in range(iterations):
            loss = self.SSE() / self.N
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            if print_loss:
                print('Loss at the',_,'iteration:',loss.item())

    def link_prediction(self):
        '''We can re-write the last line by (s_i - s_j)^t C^tX^tXC (s_i - s_j) 
         so if we construct a tensor storing S=(s_i - s_j) for all (i,j) pairs, 
         it can be written by (CS)^tX^tX(CS) and the term, X^tX, can be replaced by any kernel function K(x, x).'''
        with torch.no_grad():
            X_shape = self.X.shape
            num_samples = round(0.2 * self.N)
            idx_i_test = torch.multinomial(input=torch.arange(0, float(X_shape[0])), num_samples=num_samples,
                                        replacement=True)
            idx_j_test = torch.tensor(torch.zeros(num_samples)).long()
            for i in range(len(idx_i_test)):
                idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(X_shape[1]))[
                    torch.multinomial(input=torch.arange(idx_i_test[i].item(), float(X_shape[1])), num_samples=1,
                                    replacement=True).item()].item()  # Temp solution to sample from upper corner
            X_test = self.X.detach().clone()
            X_test[:] = 0
            X_test[idx_i_test, idx_j_test] = self.X[idx_i_test, idx_j_test]
            self.X[idx_i_test, idx_j_test] = 0  
            target = X_test[idx_i_test, idx_j_test]  # N  

            S = torch.softmax(self.S, dim=0)
            C = torch.softmax(self.C, dim=0)

            M_i = torch.matmul(torch.matmul(S, C), S[:, idx_i_test]).T #Size of test set e.g. K x N
            M_j = torch.matmul(torch.matmul(S, C), S[:, idx_j_test]).T

            #z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5 # N x N # TODO alter dist calc
            S_temp = S[:, idx_i_test].unsqueeze(1) - S[:, idx_j_test]
            CS = torch.matmul(C, S_temp)
            z_dist = CS.T@self.kernel(X_test)@CS

            theta = z_dist # N x N

            #Get the rate -> exp(log_odds) 
            rate = torch.exp(theta) # N

            fpr, tpr, threshold = metrics.roc_curve(target, rate.cpu().data.numpy())

            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target, rate.cpu().data.numpy())

            return auc_score, fpr, tpr
from numpy import zeros
import torch
import torch.nn as nn
from scipy.io import mmread
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import networkx as nx 
import seaborn as sns
import numpy as np

class Link_prediction:
    def __init__(self, N: int, A, Z, G, beta) -> None:
        self.N = N
        self.A = A # torch parameter
        self.Z = Z
        self.G = G
        self.beta = beta

    def link_prediction(self, X_test, idx_i_test, idx_j_test):
        with torch.no_grad():
            Z = F.softmax(self.Z, dim=0)
            G = F.sigmoid(self.G)
            C = (Z.T * G) / (Z.T * G).sum(0) #Gating function

            M_i = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, idx_i_test])).T #Size of test set e.g. K x N
            M_j = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, idx_j_test])).T
            z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5 # N x N 
            theta = (self.beta[idx_i_test] + self.beta[idx_j_test] - z_pdist_test) # N x N

            #Get the rate -> exp(log_odds) 
            rate = torch.exp(theta) # N

            target = self.find_target()

            fpr, tpr, threshold = metrics.roc_curve(target, rate.numpy())


            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target, rate.cpu().data.numpy())

            return auc_score, fpr, tpr

    def get_test_idx(self):
            num_samples = round(0.2 * self.N)
            idx_i_test = torch.multinomial(input=torch.arange(0, float(self.N)), num_samples=num_samples,
                                        replacement=True)
            idx_j_test = torch.tensor(np.zeros(num_samples)).long()
            for i in range(len(idx_i_test)):
                idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(self.N))[
                    torch.multinomial(input=torch.arange(idx_i_test[i].item(), float(self.N)), num_samples=1,
                                    replacement=True).item()].item()  # Temp solution to sample from upper corner

            test = torch.stack((idx_i_test,idx_j_test))
            
            return test, idx_i_test, idx_j_test
    
    def find_target(self, edge_list):
        # Have to broadcast to list, since zip will create tuples of 0d tensors.
        test, _, _ = self.get_test_idx()
        test = test.tolist()
        edge_list = edge_list.tolist()
        test = list(zip(test[0], test[1]))
        edge_list = list(zip(edge_list[0], edge_list[1]))
        return [test[i] in edge_list for i in range(len(test))]

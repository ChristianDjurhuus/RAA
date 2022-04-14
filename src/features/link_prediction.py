import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

class Link_prediction():
    def __init__(self, edge_list) -> None:
        self.edge_list = edge_list
        self.test, self.idx_i_test, self.idx_j_test = self.get_test_idx()

    def link_prediction(self):
        with torch.no_grad():
            Z = torch.softmax(self.Z, dim=0)
            G = torch.sigmoid(self.G)
            C = (Z.T * G) / (Z.T * G).sum(0) #Gating function

            M_i = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.idx_i_test])).T #Size of test set e.g. K x N
            M_j = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.idx_j_test])).T
            z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5 # N x N 
            theta = (self.beta[self.idx_i_test] + self.beta[self.idx_j_test] - z_pdist_test) # N x N

            #Get the rate -> exp(log_odds) 
            rate = torch.exp(theta) # N

            target = self.find_target()

            fpr, tpr, threshold = metrics.roc_curve(target, rate.cpu().data.numpy())

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
    
    def find_target(self):
        # Have to broadcast to list, since zip will create tuples of 0d tensors.
        test = self.test.tolist()
        edge_list = self.edge_list.tolist()
        test = list(zip(test[0], test[1]))
        edge_list = list(zip(edge_list[0], edge_list[1]))
        return [test[idx] in edge_list for idx in range(len(test))]

    def plot_auc(self):
        auc_score, fpr, tpr = self.link_prediction()
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
        plt.plot([0, 1], [0, 1],'r--', label='random')
        plt.legend(loc = 'lower right')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("RAA model")
        plt.show()

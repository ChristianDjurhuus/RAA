import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import networkx as nx


class Link_prediction():
    def __init__(self, edge_list) -> None:
        self.edge_list = edge_list
        self.target = [False]
        self.labels = ""
        while True not in self.target and self.__class__.__name__ != "KAA":
            self.test, self.idx_i_test, self.idx_j_test = self.get_test_idx()
            self.target = self.find_target()

    def link_prediction(self):
        with torch.no_grad():
            if self.__class__.__name__ == "DRRAA" or self.__class__.__name__ == "DRRAA_nre" or self.__class__.__name__ == "DRRAA_ngating":
                Z = torch.softmax(self.Z, dim=0)
                G = torch.sigmoid(self.G)
                C = (Z.T * G) / (Z.T * G).sum(0) #Gating function

                M_i = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.idx_i_test])).T #Size of test set e.g. K x N
                M_j = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.idx_j_test])).T
                z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5 # N x N 
                theta = (self.beta[self.idx_i_test] + self.beta[self.idx_j_test] - z_pdist_test) # N x N
            if self.__class__.__name__ == "LSM":
                z_pdist_test = ((self.latent_Z[self.idx_i_test,:] - self.latent_Z[-self.idx_j_test,:] + 1e-06)**2).sum(-1)**0.5 # N x N
                theta = self.beta[self.idx_i_test]+self.beta[self.idx_j_test] - z_pdist_test #(Sample_size)

            if self.__class__.__name__ == "KAA":
                X_shape = self.X.shape
                num_samples = 15
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

                S = torch.softmax(self.S, dim=0)
                C = torch.softmax(self.C, dim=0)

                M_i = torch.matmul(torch.matmul(S, C), S[:, idx_i_test]).T #Size of test set e.g. K x N
                M_j = torch.matmul(torch.matmul(S, C), S[:, idx_j_test]).T
                #M_i = torch.matmul(torch.matmul(self.X, C), S[:, idx_i_test]).T #Size of test set e.g. K x N
                #M_j = torch.matmul(torch.matmul(self.X, C), S[:, idx_j_test]).T
                z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5 # N x N 
                #z_pdist_test = torch.from_numpy(pairwise_distances(M_i, M_j, "jaccard"))
                theta = z_pdist_test # N x N

            #Get the rate -> exp(log_odds) 
            rate = torch.exp(theta) # N

            fpr, tpr, threshold = metrics.roc_curve(self.target, rate.cpu().data.numpy())

            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(self.target, rate.cpu().data.numpy())

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
        if self.__class__.__name__ == "DRRAA":
            plt.title("RAA model")
        if self.__class__.__name__ == "LSM":
            plt.title("LSM model")
        plt.show()

    def get_labels(self, attribute):
        # This only works with a gml file
        graph = nx.read_gml(self.data)
        return list(nx.get_node_attributes(graph, attribute).values())

    def get_embeddings(self):
        if self.__class__.__name__ == "DRRAA":
            Z = torch.softmax(self.Z, dim=0)
            G = torch.sigmoid(self.G)
            C = (Z.T * G) / (Z.T * G).sum(0)

            embeddings = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z)).T
            embeddings = embeddings.cpu().detach().numpy()
            archetypes = torch.matmul(self.A, torch.matmul(Z, C))
            archetypes = archetypes.cpu().detach().numpy()
            return embeddings, archetypes
        if self.__class__.__name__ == "LSM":
            return self.latent_Z.cpu().detach().numpy(), 0

    def KNeighborsClassifier(self, attribute):
        if self.labels == "":
            self.labels = self.get_labels(attribute)
        # TODO Talk about how we get the split
        X, _ = self.get_embeddings()
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(self.labels) # label encoding
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
        knn = KNeighborsClassifier(n_neighbors = 10).fit(train_X, train_y)
        return knn.score(test_X, test_y)
    
    def k_means(self, attribute, n_clusters):
        if self.labels == "":
            self.labels = self.get_lables(attribute)
        X, _ = self.get_embeddings()
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(self.labels) # label encoding
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        kmeans = KMeans(n_clusters = n_clusters, random_state = 42).fit(train_X, train_y)
        return kmeans.score(test_X, test_y)

    def logistic_regression(self, attribute):
        if self.labels == "":
            self.labels = self.get_lables(attribute)
        X, _ = self.get_embeddings()
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(self.labels) # label encoding
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
        reg = LogisticRegression(solver = "saga", max_iter = 1000, random_state = 42).fit(train_X, train_y)
        return reg.score(test_X, test_y)



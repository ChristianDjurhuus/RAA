import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
import networkx as nx
import archetypes

class Link_prediction():
    def __init__(self):
        '''
        Link prediction class. Takes self.data as edgelist format and self.G as Networkx graph format and returns
        roc_curve_AUC on a test set. Test data is 50% of the possible relationships between nodes in the graph.
        '''
        self.target = [False]
        self.labels = ""
        if self.data_type != "sparse" and self.link_pred:
            self.target, self.idx_i_test, self.idx_j_test = self.get_test_and_train()
        #while (True not in self.target or False not in self.target) and self.__class__.__name__ == "KAA":
        #    self.target, self.idx_i_test, self.idx_j_test, self.X_test = self.get_test_and_train()


    def link_prediction(self):
        with torch.no_grad():
            if self.data_type != "sparse":
                if self.__class__.__name__ == "DRRAA" or self.__class__.__name__ == "DRRAA_nre" or self.__class__.__name__ == "DRRAA_ngating":
                    Z = torch.softmax(self.Z, dim=0)
                    G = torch.sigmoid(self.Gate)
                    C = (Z.T * G) / (Z.T * G).sum(0) #Gating function

                    M_i = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.idx_i_test])).T #Size of test set e.g. K x N
                    M_j = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.idx_j_test])).T
                    z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5 # N x N 
                    theta = (self.beta[self.idx_i_test] + self.beta[self.idx_j_test] - z_pdist_test) # N x N
                if self.__class__.__name__ == "LSM":
                    z_pdist_test = ((self.latent_Z[self.idx_i_test,:] - self.latent_Z[self.idx_j_test,:] + 1e-06)**2).sum(-1)**0.5 # N x N
                    theta = self.beta[self.idx_i_test]+self.beta[self.idx_j_test] - z_pdist_test #(Sample_size)
                if self.__class__.__name__ == 'LSMAA':
                    # Do the AA on the lsm embeddings
                    aa = archetypes.AA(n_archetypes=self.k)
                    lsm_z = aa.fit_transform(self.latent_Z.detach().numpy())
                    latent_Z = torch.from_numpy(lsm_z).float()
                    z_pdist_test = ((latent_Z[self.idx_i_test, :] - latent_Z[self.idx_j_test, :] + 1e-06) ** 2).sum(
                        -1) ** 0.5  # N x N
                    theta = self.beta - z_pdist_test  # (Sample_size)

                if self.__class__.__name__ == "KAA":
                    S = torch.softmax(self.S, dim=0)
                    C = torch.softmax(self.C, dim=0)

                    CtKC = C.T @ self.kernel(self.X_test,
                                            type=self.type) @ C
                    z_dist = torch.zeros(len(self.idx_i_test))
                    for i in range(len(self.idx_i_test)):
                        z_dist[i] = (S[:, self.idx_i_test[i]].T @ CtKC @ S[:, self.idx_i_test[i]]
                                + S[:, self.idx_j_test[i]].T @ CtKC @ S[:, self.idx_j_test[i]]
                                - 2*(S[:, self.idx_i_test[i]].T @ CtKC @ S[:, self.idx_j_test[i]]))+1e-06
                    theta = -z_dist  #(test_size)
            
            if self.data_type == "sparse":
                # Create targets
                self.target = torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]), torch.ones(self.sparse_i_idx_removed.shape[0])))

                if self.__class__.__name__ == "DRRAA" or self.__class__.__name__ == "DRRAA_nre" or self.__class__.__name__ == "DRRAA_ngating":
                    Z = torch.softmax(self.Z, dim=0)
                    G = torch.sigmoid(self.Gate)
                    C = (Z.T * G) / (Z.T * G).sum(0) #Gating function

                    M_i = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.removed_i])).T #Size of test set e.g. K x N
                    M_j = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z[:, self.removed_j])).T
                    z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5 # N x N 
                    theta = (self.beta[self.removed_i] + self.beta[self.removed_j] - z_pdist_test) # N x N
                if self.__class__.__name__ == "LSM":
                    z_pdist_test = ((self.latent_Z[self.removed_i,:] - self.latent_Z[self.removed_j,:] + 1e-06)**2).sum(-1)**0.5 # N x N
                    theta = self.beta[self.removed_i] + self.beta[self.removed_j] - z_pdist_test #(Sample_size)
                if self.__class__.__name__ == 'LSMAA':
                    # Do the AA on the lsm embeddings
                    aa = archetypes.AA(n_archetypes=self.k)
                    lsm_z = aa.fit_transform(self.latent_Z.detach().numpy())
                    latent_Z = torch.from_numpy(lsm_z).float()
                    z_pdist_test = ((latent_Z[self.removed_i, :] - latent_Z[self.removed_j, :] + 1e-06) ** 2).sum(
                        -1) ** 0.5  # N x N
                    theta = self.beta - z_pdist_test  # (Sample_size)

                if self.__class__.__name__ == "KAA":
                    S = torch.softmax(self.S, dim=0)
                    C = torch.softmax(self.C, dim=0)

                    CtKC = C.T @ self.kernel(self.X_test,
                                            type=self.type) @ C
                    z_dist = torch.zeros(len(self.removed_i))
                    for i in range(len(self.removed_i)):
                        z_dist[i] = (S[:, self.removed_i[i]].T @ CtKC @ S[:, self.removed_i[i]]
                                + S[:, self.removed_j[i]].T @ CtKC @ S[:, self.removed_j[i]]
                                - 2*(S[:, self.removed_i[i]].T @ CtKC @ S[:, self.removed_j[i]]))+1e-06
                    theta = -z_dist  #(test_size)

            #Get the rate -> exp(log_odds) 
            rate = torch.exp(theta) # N

            fpr, tpr, threshold = metrics.roc_curve(self.target, rate.cpu().data.numpy())

            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(self.target, rate.cpu().data.numpy())
            print(self.__class__.__name__,':', auc_score)
            #print(self.__class__.__name__,':', f1_score(self.target, rate.cpu().data.numpy()))
            return auc_score, fpr, tpr

    def ideal_prediction(self, A, Z, beta=None):
        '''
        A: Arcetypes
        Z: sampled datapoints
        '''
        with torch.no_grad():
            if beta == None:
                beta = torch.ones(self.N)
            M_i = torch.matmul(A, Z[:, self.idx_i_test]).T
            M_j = torch.matmul(A, Z[:, self.idx_j_test]).T
            z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5
            theta = (self.beta[self.idx_i_test] + self.beta[self.idx_j_test] - z_pdist_test)
            rate = torch.exp(theta)
            fpr, tpr, threshold = metrics.roc_curve(self.target, rate.cpu().data.numpy())
            auc_score = metrics.roc_auc_score(self.target, rate.cpu().data.numpy())
            return auc_score, fpr, tpr

    def th_delete(self, tensor, indices):
        '''
        deletes elements from a tensor by index. Borrowed from Alexey_Stolpovskiy at:
        https://discuss.pytorch.org/t/how-to-remove-an-element-from-a-1-d-tensor-by-index/23109/14
        '''
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices] = False
        return tensor[mask]

    def get_test_and_train(self):
        cc_problem = False
        num_samples = round(self.test_size * (0.5*(self.N*(self.N-1))))
        target = []
        G = self.G.copy()
        idx_i_test = torch.multinomial(input=torch.arange(0, float(self.N)), num_samples=num_samples,
                                    replacement=True)
        idx_j_test = torch.tensor(np.zeros(num_samples)).long()
        delete_idx = []
        for i in range(len(idx_i_test)):
            idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(self.N))[
                torch.multinomial(input=torch.arange(idx_i_test[i].item(), float(self.N)), num_samples=1,
                                replacement=True).item()].item() # Temp solution to sample from upper corner
            target_nodes = G.neighbors(int(idx_i_test[i]))
            if int(idx_j_test[i]) in target_nodes: #Loop through neighbors (super fast instead of self.edge_list)
                G.remove_edge(int(idx_i_test[i]), int(idx_j_test[i]))
                if nx.number_connected_components(G) == 1:
                    target.append(1)
                else:
                    G.add_edge(int(idx_i_test[i]), int(idx_j_test[i])) #skip the draw if the link splits network into two components
                    cc_problem = True
                    delete_idx.append(i)
                    continue
            else:
                target.append(0)
        idx_i_test = self.th_delete(idx_i_test, delete_idx)
        idx_j_test = self.th_delete(idx_j_test, delete_idx)
        temp = [x for x in nx.generate_edgelist(G, data=False)]
        edge_list = np.zeros((2, len(temp)))
        for idx in range(len(temp)):
            edge_list[0, idx] = temp[idx].split()[0]
            edge_list[1, idx] = temp[idx].split()[1]
        self.edge_list = torch.from_numpy(edge_list).long()
        self.G = G
        if cc_problem:
            print(f'''There was a problem when removing links from the train set which could have resulted in splitting
            the network into multiple components. We decided not to remove these links, however, the test set will be
            smaller and sparser than anticipated. It now contains {int(sum(target))} edges and a sparsity of 
            {round(np.mean(target)*100,2)}% as opposed to the train set's {self.G.number_of_edges()} edges and sparsity of
            {round((self.G.number_of_edges() / ((self.G.number_of_nodes()**2))*0.5)*100,2)}% - this is after removing
            edges drawn into the test set. To avoid this, you could try to create a test and train split yourself.''')
        if self.__class__.__name__ == 'KAA':
            self.data = torch.from_numpy(nx.to_numpy_array(self.G)).long()
            X_test = self.data.clone()
            X_test[:] = 0
            X_test[idx_i_test, idx_j_test] = self.data[idx_i_test, idx_j_test]
            self.data[idx_i_test, idx_j_test] = 0
            self.X_test = X_test
            return target, idx_i_test, idx_j_test
        return target, idx_i_test, idx_j_test

    def KAA_test_train(self):
        '''We can re-write the last line by (s_i - s_j)^t C^tX^tXC (s_i - s_j)
         so if we construct a tensor storing S=(s_i - s_j) for all (i,j) pairs,
         it can be written by (CS)^tX^tX(CS) and the term, X^tX, can be replaced by any kernel function K(x, x).'''
        X_shape = self.data.shape
        num_samples = round(self.test_size * self.N)
        idx_i_test = torch.multinomial(input=torch.arange(0, float(X_shape[0])),
                                       num_samples=num_samples,
                                       replacement=True)
        idx_j_test = torch.tensor(torch.zeros(num_samples)).long()
        for i in range(len(idx_i_test)):
            idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(X_shape[1]))[
                torch.multinomial(input=torch.arange(idx_i_test[i].item(), float(X_shape[1])),
                                  num_samples=1,
                                  replacement=True).item()].item()  # Temp solution to sample from upper corner
        X_test = self.data.detach().clone()
        X_test[:] = 0
        X_test[idx_i_test, idx_j_test] = self.data[idx_i_test, idx_j_test]
        self.data[idx_i_test, idx_j_test] = 0
        target = X_test[idx_i_test, idx_j_test]  # N
        return target, idx_i_test, idx_j_test, X_test

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
        return list(nx.get_node_attributes(self.G, attribute).values())

    def get_embeddings(self):
        # TODO extend this to all the other classes.
        if self.__class__.__name__ == "DRRAA":
            Z = torch.softmax(self.Z, dim=0)
            G = torch.sigmoid(self.Gate)
            C = (Z.T * G) / (Z.T * G).sum(0)

            embeddings = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z)).T
            embeddings = embeddings.cpu().detach().numpy()
            archetypes = torch.matmul(self.A, torch.matmul(Z, C))
            archetypes = archetypes.cpu().detach().numpy()
            return embeddings, archetypes
        if self.__class__.__name__ == "LSM":
            return self.latent_Z.cpu().detach().numpy(), 0

    def KNeighborsClassifier(self, attribute, n_neighbors = 10):
        self.labels = self.get_labels(attribute)
        X, _ = self.get_embeddings()
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(self.labels) # label encoding
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
        knn = KNeighborsClassifier(n_neighbors = n_neighbors).fit(train_X, train_y)
        return knn.score(test_X, test_y)
    
    def k_means(self, attribute, n_clusters = 10):
        self.labels = self.get_labels(attribute)
        X, _ = self.get_embeddings()
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(self.labels) # label encoding
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        kmeans = KMeans(n_clusters = n_clusters, random_state = 42).fit(train_X, train_y)
        return kmeans.score(test_X, test_y)

    def logistic_regression(self, attribute):
        self.labels = self.get_labels(attribute)
        X, _ = self.get_embeddings()
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(self.labels) # label encoding
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
        reg = LogisticRegression(solver = "saga", max_iter = 1000, random_state = 42).fit(train_X, train_y)
        return reg.score(test_X, test_y)



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
import archetypes
import random
import time


class Link_prediction():
    def __init__(self):
        '''
        Link prediction class. Takes self.data as edgelist format and self.G as Networkx graph format and returns
        roc_curve_AUC on a test set. Test data is 50% of the possible relationships between nodes in the graph.
        '''
        self.target = [False]
        self.labels = ""
        while (True not in self.target or False not in self.target) and self.__class__.__name__ != "KAA":
            self.target, self.idx_i_test, self.idx_j_test = self.get_test_and_train() 



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

                S = torch.softmax(self.S, dim=0)
                C = torch.softmax(self.C, dim=0)

                M_i = torch.matmul(torch.matmul(S, C), S[:, idx_i_test]).T #Size of test set e.g. K x N
                M_j = torch.matmul(torch.matmul(S, C), S[:, idx_j_test]).T

                z_pdist_test = ((M_i - M_j + 1e-06)**2).sum(-1)**0.5 # N x N 

                theta = z_pdist_test # N x N

            #Get the rate -> exp(log_odds) 
            rate = torch.exp(theta) # N

            fpr, tpr, threshold = metrics.roc_curve(self.target, rate.cpu().data.numpy())

            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(self.target, rate.cpu().data.numpy())
            print(self.__class__.__name__,':', auc_score)
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


    def get_test_and_train_temp(self):
        #TODO: Might have issues with sampling from lower and upper triangle
        #Positive links (50% of edges) and negativeg links (Equal # as positive)
        #The heavy memory users:
        n_components_train = 2
        while 1 < n_components_train:
            t0 = time.time()
            G_altered = self.G.copy()
            edges = list(self.G.edges)
            nonedges = list(nx.non_edges(self.G))
            chosen_edge = random.sample(edges, round(0.5 * len(edges))) 
            N_chosen_edge = len(chosen_edge)
            chosen_nonedge = random.sample(nonedges, N_chosen_edge)
            target = [True] * N_chosen_edge + [False] * N_chosen_edge
            #Need to be optimized
            idx_i_test = torch.tensor(np.zeros(N_chosen_edge*2)).long()
            idx_j_test = torch.tensor(np.zeros(N_chosen_edge*2)).long()

            #Ensure that we sample upper triangle
            for i in range(N_chosen_edge):
                #positive links
                edge = chosen_edge[i]
                if edge[0] < edge[1]:
                    idx_i_test[i] = edge[0]
                    idx_j_test[i] = edge[1]
                else:
                    idx_i_test[i] = edge[1]
                    idx_j_test[i] = edge[0]
                #negative links
                nonedge = chosen_nonedge[i]
                if nonedge[0] < nonedge[1]:
                    idx_i_test[i + N_chosen_edge] = nonedge[0]
                    idx_j_test[i + N_chosen_edge] = nonedge[1]
                else:
                    idx_i_test[i + N_chosen_edge] = nonedge[1]
                    idx_j_test[i + N_chosen_edge] = nonedge[0]
            t1 = time.time()
            print(f'The time it took to generate train test split: {t1-t0}')

            G_altered.remove_edges_from(chosen_edge)
            n_components_train = nx.number_connected_components(G_altered)

        temp = [x for x in nx.generate_edgelist(G_altered, data=False)]
        edge_list = np.zeros((2, len(temp)))
        for idx in range(len(temp)):
            edge_list[0, idx] = temp[idx].split()[0]
            edge_list[1, idx] = temp[idx].split()[1]
        self.edge_list = torch.from_numpy(edge_list).long()
        self.G = G_altered

        return target, idx_i_test, idx_j_test



    def get_test_and_train(self):
        num_samples = round(0.5 * (0.5*(self.N*(self.N-1))))
        g_num_edge_init = self.G.number_of_edges()
        n_components_train = 2
        target = np.zeros(num_samples)
        while 1 < n_components_train: #Make sure to never remove links so we get more than one component
            G = self.G.copy()
            t0 = time.time()
            idx_i_test = torch.multinomial(input=torch.arange(0, float(self.N)), num_samples=num_samples,
                                        replacement=True)
            idx_j_test = torch.tensor(np.zeros(num_samples)).long()
            for i in range(len(idx_i_test)):
                idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(self.N))[
                    torch.multinomial(input=torch.arange(idx_i_test[i].item(), float(self.N)), num_samples=1,
                                    replacement=True).item()].item() # Temp solution to sample from upper corner
                target_nodes = G.neighbors(int(idx_i_test[i]))
                if int(idx_j_test[i]) in target_nodes: #Loop through neighbors (super fast instead of self.edge_list)
                    G.remove_edge(int(idx_i_test[i]), int(idx_j_test[i]))
                    target[i] = 1
            t1 = time.time()
            print(f'The time it took to generate train test split: {t1-t0}')
            n_components_train = nx.number_connected_components(G)
        temp = [x for x in nx.generate_edgelist(G, data=False)]
        edge_list = np.zeros((2, len(temp)))
        for idx in range(len(temp)):
            edge_list[0, idx] = temp[idx].split()[0]
            edge_list[1, idx] = temp[idx].split()[1]
        self.edge_list = torch.from_numpy(edge_list).long()
        self.G = G
        print(f'The proportion of removed edges: {self.G.number_of_edges() / g_num_edge_init}')
        print(f'The propotion of negative examples: {sum(np.where(target==0))/len(target)}')
        return target, idx_i_test, idx_j_test

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



#from cv2 import arcLength
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from matplotlib import patches
from src.data.synthetic_data import truncate_colormap
#from fast_histogram import histogram2d

class Visualization():
    def __init__(self) -> None:
        pass

    def get_embeddings(self):
        if self.__class__.__name__ == "DRRAA":
            Z = torch.softmax(self.Z, dim=0)
            G = torch.sigmoid(self.Gate)
            C = (Z.T * G) / (Z.T * G).sum(0)
            u, sigma, v = torch.svd(self.A) # Decomposition of A.
            r = torch.matmul(torch.diag(sigma), v.T)
            embeddings = torch.matmul(r, torch.matmul(torch.matmul(Z, C), Z)).T
            archetypes = torch.matmul(r, torch.matmul(Z, C))
            embeddings = embeddings.cpu().detach().numpy()
            archetypes = archetypes.cpu().detach().numpy()
            return embeddings, archetypes

        if self.__class__.__name__ == "LSM":
            return self.latent_Z.cpu().detach().numpy(), 0

        if self.__class__.__name__ == "KAA":
            S = torch.softmax(self.S, dim=0)
            C = torch.softmax(self.C, dim=0)
            embeddings = (torch.matmul(torch.matmul(S, C), S).T).cpu().detach().numpy()
            archetypes = torch.matmul(S, C).cpu().detach().numpy()
            return embeddings, archetypes
        
        if self.__class__.__name__ == "BDRRAA":
            Z_i = torch.softmax(self.Z_i, dim=0)
            Z_j = torch.softmax(self.Z_j, dim=0)
            Z = torch.cat((Z_i,Z_j),1)
            G = torch.sigmoid(self.Gate)
            C = (Z.T * G) / (Z.T * G).sum(0)

            embeddings = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z)).T
            archetypes = torch.matmul(self.A, torch.matmul(Z, C))
            return embeddings, archetypes

    def plot_latent_and_loss(self, iterations, cmap='red', file_name=None):
        embeddings, archetypes = self.get_embeddings()
        if self.__class__.__name__ == "DRRAA" or self.__class__.__name__ == "KAA":
            if embeddings.shape[1] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(embeddings[:, 0], embeddings[:, 1],
                        embeddings[:, 2], c='red')
                ax.scatter(archetypes[0, :], self.archetypes[1, :],
                        self.archetypes[2, :], marker='^', c='black')
            else:
                fig = plt.subplots(dpi=500)
                if type(cmap) == dict:
                    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=list(cmap.values()), cmap="tab10", label="Node embeddings")
                else:
                    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cmap, cmap="tab10", label="Node embeddings")

                plt.scatter(archetypes[0, :], archetypes[1, :], marker='^', c='black', label="Archetypes")
                # Plotting learning curve
                '''if self.__class__.__name__ == "KAA":
                    ax2.plot(self.losses, c="#F2D42E")
                    ax2.set_yscale("log")
                else:    
                    ax2.plot(self.losses, c="#C4000D")'''
            if file_name != None:        
                plt.savefig(file_name, dpi=500)
            else:
                plt.show()

        if self.__class__.__name__ == "LSM":
            if embeddings.shape[1] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(embeddings[:, 0], embeddings[:, 1],
                        embeddings[:, 2], c='red')
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, dpi=500)
                if type(cmap) == dict:
                    ax1.scatter(embeddings[:, 0], embeddings[:, 1], c=list(cmap.values()), cmap="Set2", label="Node embeddings")
                else:
                    ax1.scatter(embeddings[:, 0], embeddings[:, 1], c=cmap, cmap="Set2", label="Node embeddings")
                ax1.legend()
                # Plotting learning curve
                ax2.plot(self.losses, c="#00C700")
                ax2.set_yscale("log")
            plt.savefig(file_name, dpi=500)
            #plt.show()

        if self.__class__.__name__ == "BDRRAA":
            embeddings, archetypes = self.get_embeddings()
            if embeddings.shape[1] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(embeddings[:, 0].detach().numpy(), embeddings[:, 1].detach().numpy(),
                        embeddings[:, 2].detach().numpy(), c='red')
                ax.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(),
                        archetypes[2, :].detach().numpy(), marker='^', c='black')
                plt.show()

            else:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.scatter(embeddings[self.sample_shape[0]:, 0].detach().numpy(), embeddings[self.sample_shape[0]:, 1].detach().numpy(), c='red')
                ax1.scatter(embeddings[:self.sample_shape[0], 0].detach().numpy(), embeddings[:self.sample_shape[0], 1].detach().numpy(), c='blue')
                ax1.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(), marker='^', c='black')
                plt.show()



    def plot_loss(self):
        fig, ax = plt.subplots(dpi = 100)
        ax.plot(self.losses, c="#00C700")
        ax.set_yscale("log")
        plt.show()

    def embedding_density(self, filename = "embedding_density.pdf", show = False):
        embeddings, archetypes = self.get_embeddings()
        x_bins = np.linspace(embeddings[:, 0].min(), embeddings[:, 0].max(), round(np.sqrt(self.N)))
        y_bins = np.linspace(embeddings[:, 1].min(), embeddings[:, 1].max(), round(np.sqrt(self.N)))
        cmap = plt.get_cmap('RdPu')
        cmap = truncate_colormap(cmap, 0.2, 1)
        plt.hist2d(embeddings[:, 0], embeddings[:, 1], cmap = cmap, bins = [x_bins, y_bins], norm=colors.LogNorm())
        plt.colorbar()
        plt.savefig(filename, dpi = 500)
        if show:
            plt.show()
        else:
            plt.clf()

    def get_labels(self, attribute):
        # This only works with a gml file
        return list(nx.get_node_attributes(self.G, attribute).values())

    def archetypal_nodes(self):
        embeddings, archetypes = self.get_embeddings()
        closest_node = np.zeros(archetypes.shape[1])
        for i in range(archetypes.shape[1]):
            closest_node[i] = np.argmin(((embeddings - archetypes[:,i]) ** 2).sum(-1))

        return closest_node

    def archetype_partitions(self):
        Z = torch.softmax(self.Z, dim=0)
        partitions =[[] for i in range(Z.shape[0])]
        for i in range(Z.shape[1]):
            idx = torch.argmax(Z[:,i])
            partitions[idx].append(i)
        return partitions


    def order_adjacency_matrix(self, filename="ordered_adj_m.png", show = True):
        """
        - G is a netorkx graph
        - node_order (optional) is a list of nodes, where each node in G
            appears exactly once
        - partitions is a list of node lists, where each node in G appears
            in exactly one node list
        - colors is a list of strings indicating what color each
            partition should be
        If partitions is specified, the same number of colors needs to be
        specified.
        """
        if self.data_type == "sparse":
            # Collect the entire graph
            i_partion = np.concatenate((self.sparse_i_idx.cpu(), self.sparse_i_idx_removed.cpu()))
            j_partion = np.concatenate((self.sparse_j_idx.cpu(), self.sparse_j_idx_removed.cpu()))
            edge_list = np.zeros((2, len(i_partion)))
            for idx in range(len(i_partion)):
                edge_list[0, idx] = i_partion[idx]
                edge_list[1, idx] = j_partion[idx]
            edge_list = list(zip(edge_list[0], edge_list[1]))
            
            # Make graph
            self.G = nx.from_edgelist(edge_list)

        colors=["blue"]
        partitions  = [self.archetype_partitions()]
        node_order = [node for archetype in partitions[0] for node in archetype]
        adjacency_matrix = nx.to_numpy_matrix(self.G, nodelist=node_order)

        #Plot adjacency matrix in toned-down black and white
        fig = plt.figure(figsize=(5, 5), dpi = 200) # in inches
        plt.imshow(adjacency_matrix,
                    cmap="Greys",
                    interpolation="none")
        
        # The rest is just if you have sorted nodes by a partition and want to
        # highlight the module boundaries
        assert len(partitions) == len(colors)
        ax = plt.gca()
        for partition, color in zip(partitions, colors):
            current_idx = 0
            for module in partition:
                ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                            len(module), # Width
                                            len(module), # Height
                                            facecolor="none",
                                            edgecolor=color,
                                            linewidth="1"))
                current_idx += len(module)
        fig.savefig(filename, dpi=500)

        if show:
            plt.show()
        else:
            plt.clf()


    def decision_boundary_linear(self, attribute, ax=None):
        """
        Now only works for binary classes
        https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/
        """
        self.labels = self.get_labels(attribute)
        # TODO Talk about how we get the split
        X, _ = self.get_embeddings()
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(self.labels) # label encoding
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
        reg = LogisticRegression(solver = "saga", max_iter = 1000, random_state = 42).fit(train_X, train_y)
        reg.fit(train_X, train_y)

        # Retrieve the model parameters.
        b = reg.intercept_[0]
        w1, w2 = reg.coef_.T
        # Calculate the intercept and gradient of the decision boundary.
        c = -b/w2
        m = -w1/w2

        # Plot the data and the classification with the decision boundary.
        xmin, xmax = X[:, 0].min(), X[:, 0].max()
        ymin, ymax = X[:, 1].min(), X[:, 1].max()
        xd = np.array([xmin, xmax])
        yd = m * xd + c


        #fig, ax = plt.subplots(dpi = 500)
        ax.set_xlim(left = xmin, right = xmax)
        ax.set_ylim(bottom = ymin, top = ymax)
        ax.plot(xd, yd, 'k', lw = 1, ls = '--')
        ax.fill_between(xd, yd, ymin, color='tab:orange', alpha=0.2)
        ax.fill_between(xd, yd, ymax, color='tab:blue', alpha=0.2)
        ax.scatter(*X[y == 0].T, s = 8, alpha=0.5)
        ax.scatter(*X[y == 1].T, s = 8, alpha=0.5)
        ax.set_ylabel(r'$x_2$', fontsize = "medium")
        ax.set_xlabel(r'$x_1$', fontsize = "medium")
        ax.set_title("Decision Boundary - LR", fontsize = "large")

        #fig.savefig("Desicion_boundary_LR.png")
        #plt.show()

        return ax #reg.score(test_X, test_y)

    def decision_boundary_knn(self, attribute, n_neighbors = 10, filename=False): #TODO test if this works
        # https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib
        self.labels = self.get_labels(attribute)
        # TODO Talk about how we get the split
        X, archetypes = self.get_embeddings()
        #X = X+self.beta.unsqueeze(1).detach().numpy()
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(self.labels) # label encoding
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)
        knn = KNeighborsClassifier(n_neighbors = n_neighbors).fit(train_X, train_y)

        h = .02  # step size in the mesh
        # Create color maps


        cmap_light = ListedColormap(['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'])
        cmap_bold = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots(dpi = 500)
        ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        ax.scatter(archetypes[0, :], archetypes[1, :], marker='^', c='black', label="Archetypes")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        if not filename:
            fig.savefig("Desicion_boundary_KNN.png",dpi=500)
        else:
            fig.savefig(filename, dpi=500)
        #plt.show()

        return knn.score(test_X, test_y)


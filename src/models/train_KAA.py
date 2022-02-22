from numpy import zeros
import torch
import torch.nn as nn
from scipy.io import mmread
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import networkx as nx


class KAA(nn.Module):
    def __init__(self, A, input_size, k):
        super(KAA, self).__init__()
        self.A = A
        self.input_size = input_size
        self.k = k

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.a = torch.nn.Parameter(torch.randn(1))

        self.X = torch.nn.Parameter(torch.randn(self.k, self.input_size[0]))
        self.C = torch.nn.Parameter(torch.randn(self.input_size[0], self.k))


    def random_sampling(self):
        # TODO

        return None

    def kernel(self, xt, x):

        kernel = xt@x 


        return kernel

    def log_likelihood(self):
        '''We can re-write the last line by (s_i - s_j)^t C^tX^tXC (s_i - s_j) 
        so if we construct a tensor storing S=(s_i - s_j) for all (i,j) pairs, 
        it can be written by (CS)^tX^tX(CS) and the term, X^tX, can be replaced by any kernel function K(x, x).'''

        beta = self.beta.unsqueeze(1) + self.beta  # (N x N)
        X = F.softmax(self.X, dim=0)  # (K x N)
        C = F.softmax(self.C, dim=0)  # (N x K)
        z_dist = torch.tensor(zeros(self.input_size))
        for i in range(self.X.shape[1]):
            for j in range(self.X.shape[1]):
                S = self.X[:,i] - self.X[:,j]
                CS = C@S
                z_dist[i, j] = CS.T@self.kernel(X.T,X)@CS

        theta = beta - self.a * z_dist  # (N x N)
        softplus_theta = F.softplus(theta)  # log(1+exp(theta))
        LL = 0.5 * (theta * self.A).sum() - 0.5 * torch.sum(
            softplus_theta - torch.diag(torch.diagonal(softplus_theta)))  # Times by 0.5 to avoid double counting

        return LL

    def link_prediction(self, A_test, idx_i_test, idx_j_test):
        with torch.no_grad():
            X = F.softmax(self.X, dim=0)
            C = F.softmax(self.C, dim=0)

            M_i = torch.matmul(torch.matmul(X, C), X[:, idx_i_test]).T  # Size of test set e.g. K x N
            M_j = torch.matmul(torch.matmul(X, C), X[:, idx_j_test]).T
            


            z_pdist_test = ((M_i.unsqueeze(1) - M_j + 1e-06) ** 2).sum(-1) ** 0.5  # N x N
            theta = (self.beta[idx_i_test] + self.beta[idx_j_test] - self.a * z_pdist_test)  # N x N

            # Get the rate -> exp(log_odds)
            rate = torch.exp(theta).flatten()  # N^2

            # Create target (make sure its in the right order by indexing)
            target = A_test[idx_i_test.unsqueeze(1), idx_j_test].flatten()  # N^2

            fpr, tpr, threshold = metrics.roc_curve(target.numpy(), rate.numpy())

            # Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target.cpu().data.numpy(), rate.cpu().data.numpy())

            return auc_score, fpr, tpr


if __name__ == "__main__":
    seed = 1984
    torch.random.manual_seed(seed)

    # A = mmread("data/raw/soc-karate.mtx")
    # A = A.todense()
    ZKC_graph = nx.karate_club_graph()
    # Let's keep track of which nodes represent John A and Mr Hi
    Mr_Hi = 0
    John_A = 33

    # Let's display the labels of which club each member ended up joining
    club_labels = nx.get_node_attributes(ZKC_graph, 'club')

    # Getting adjacency matrix
    A = nx.convert_matrix.to_numpy_matrix(ZKC_graph)
    A = torch.from_numpy(A)
    k = 2

    link_pred = False

    if link_pred:
        A_shape = A.shape
        num_samples = 15
        idx_i_test = torch.multinomial(input=torch.arange(0, float(A_shape[0])), num_samples=num_samples,
                                       replacement=True)
        idx_j_test = torch.tensor(zeros(num_samples)).long()
        for i in range(len(idx_i_test)):
            idx_j_test[i] = torch.arange(idx_i_test[i].item(), float(A_shape[1]))[
                torch.multinomial(input=torch.arange(idx_i_test[i].item(), float(A_shape[1])), num_samples=1,
                                  replacement=True).item()].item()  # Temp solution to sample from upper corner

        # idx_j_test = torch.multinomial(input=torch.arange(0, float(A_shape[1])), num_samples=num_samples,
        #                               replacement=True)

        A_test = A.detach().clone()
        A_test[:] = 0
        A_test[idx_i_test, idx_j_test] = A[idx_i_test, idx_j_test]
        A[idx_i_test, idx_j_test] = 0

    model = KAA(A=A, input_size=A.shape, k=k) #Is it here we determine the kernel?
    optimizer = torch.optim.Adam(params=model.parameters())

    losses = []
    iterations = 10000
    for _ in range(iterations):
        loss = - model.log_likelihood() / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Loss at the', _, 'iteration:', loss.item())

    # Link prediction
    if link_pred:
        auc_score, fpr, tpr = model.link_prediction(A_test, idx_i_test, idx_j_test)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc_score)
        plt.plot([0, 1], [0, 1], 'r--', label='random')
        plt.legend(loc='lower right')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("RAA model")
        plt.show()

    # Plotting latent space
    X = F.softmax(model.X, dim=0)
    C = F.softmax(model.C, dim=0)
    embeddings = torch.matmul(torch.matmul(X, C), X).T
    archetypes = torch.matmul(X, C)

    labels = list(club_labels.values())
    idx_hi = [i for i, x in enumerate(labels) if x == "Mr. Hi"]
    idx_of = [i for i, x in enumerate(labels) if x == "Officer"]

    if embeddings.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(embeddings[:, 0].detach().numpy()[idx_hi], embeddings[:, 1].detach().numpy()[idx_hi],
                   embeddings[:, 2].detach().numpy()[idx_hi], c='red', label='Mr. Hi')
        ax.scatter(embeddings[:, 0].detach().numpy()[idx_of], embeddings[:, 1].detach().numpy()[idx_of],
                   embeddings[:, 2][idx_of].detach().numpy(), c='blue', label='Officer')
        ax.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(),
                   archetypes[2, :].detach().numpy(), marker='^', c='black')
        ax.text(embeddings[Mr_Hi, 0].detach().numpy(), embeddings[Mr_Hi, 1].detach().numpy(),
                embeddings[Mr_Hi, 2].detach().numpy(), 'Mr. Hi')
        ax.text(embeddings[John_A, 0].detach().numpy(), embeddings[John_A, 1].detach().numpy(),
                embeddings[John_A, 2].detach().numpy(), 'Officer')
        ax.set_title(f"Latent space after {iterations} iterations")
        ax.legend()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.scatter(embeddings[:, 0].detach().numpy()[idx_hi], embeddings[:, 1].detach().numpy()[idx_hi], c='red',
                    label='Mr. Hi')
        ax1.scatter(embeddings[:, 0].detach().numpy()[idx_of], embeddings[:, 1].detach().numpy()[idx_of], c='blue',
                    label='Officer')
        ax1.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(), marker='^', c='black')
        ax1.annotate('Mr. Hi', embeddings[Mr_Hi, :])
        ax1.annotate('Officer', embeddings[John_A, :])
        ax1.legend()
        ax1.set_title(f"Latent space after {iterations} iterations")
        # Plotting learning curve
        ax2.plot(losses)
        ax2.set_title("Loss")
    plt.show()

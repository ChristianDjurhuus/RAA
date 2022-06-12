
from os import link
from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
from src.models.train_BDRRAA_module import BDRRAA
from src.models.train_KAAsparse_module import KAAsparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import seaborn as sns

def main():
    seed = 4
    torch.random.manual_seed(seed)
    # Data and hyperparameters
    data = 'data/raw/karate/karate.gml'
    node_attribute = "club"
    k = 2
    d = 2

    # Model
    #model = DRRAA(data = data,
    #                k=k,
    #                d=d,
    #                data_type = "gml", 
    #                sample_size=0.5) # Set sampling procentage size
    scores = []
    for q in range(10):
        torch.manual_seed(q)
        model = LSM(d = d,
                        sample_size = 1,
                        data = data,
                        data_type = "gml", link_pred = True)
        # Train
        iterations = 10
        model.train(iterations = iterations, LR = 0.01, print_loss = True)
        # Visualization
        #model.plot_latent_and_loss(iterations)
        #model.embedding_density()
        #knn_score = model.KNeighborsClassifier(attribute = node_attribute)
        #print(f"knn_score: {knn_score}")
        #log_reg = model.logistic_regression(attribute = node_attribute)
        #print(f"logistic regression score: {log_reg}")
        # Link prediction
        score, _, _ = model.link_prediction()
        print(score)
        scores.append((score,np.mean(model.losses[-100:])))
    return scores
    
def main2():
    seed = 4
    torch.random.manual_seed(seed)
    # Data and hyperparameters
    data = 'data/raw/karate/karate.gml'
    node_attribute = "club"
    k = 3
    d = 2

    # Model
    model = DRRAA(data = data,
                    k = k,
                    d = d,
                    data_type = "gml", 
                    sample_size = 1, link_pred = True) # Set sampling procentage size
    #model = LSM(latent_dim = d,
    #                sample_size = 1,
    #                data = data,
    #                data_type = "gml")
    iterations = 30
    model.train(iterations = iterations, LR = 0.01, print_loss = True)
    model.plot_latent_and_loss(iterations = iterations)
    model.embedding_density()
    model.plot_latent_and_loss(iterations)
    model.embedding_density()
    knn_score = model.KNeighborsClassifier(attribute = node_attribute)
    print(f"knn_score: {knn_score}")
    log_reg = model.logistic_regression(attribute = node_attribute)
    print(f"logistic regression score: {log_reg}")

    model.decision_boundary_linear(attribute = node_attribute)
    model.decision_boundary_knn(attribute = node_attribute)
    model.plot_auc()


def main3():
    seed = 4
    torch.random.manual_seed(seed)
    # Data and hyperparameters
    dataset = "facebook"
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    k = 8
    d = 2

    # Model
    model = DRRAA(data = data,
                    data_2 = data2,
                    k=k,
                    d=d,
                    data_type = "sparse",
                    sample_size=0.1,
                    non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem) # Set sampling procentage size

    iterations = 2000
    model.train(iterations = iterations, LR = 0.1, print_loss = True)
    model.order_adjacency_matrix(filename = f"facebook_ordered.pdf", show = True)
    #model.plot_latent_and_loss(iterations = iterations)
    #model.embedding_density()
    #model.plot_auc()
    #model.decision_boundary(attribute = node_attribute)

def main4():
    seed = 4
    torch.random.manual_seed(seed)
    # Data and hyperparameters
    dataset = "hepth"
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    d = 2
    k = 8

    # Model
    model = DRRAA(data = data,
                    data_2 = data2,
                    k = k,
                    d = d,
                    data_type = "sparse",
                    sample_size=0.5,
                    link_pred = True,
                    non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem) # Set sampling procentage size

    iterations = 10000
    model.train(iterations = iterations, LR = 0.1, print_loss = True, scheduling = True)
    model.plot_latent_and_loss(iterations = iterations)
    model.embedding_density()
    model.plot_auc()
    score, _, _ = model.link_prediction()
    print(score)

def main5():
    seed = 42
    torch.random.manual_seed(seed)
    k = 3
    d = 2

    # Data
    dataset = "drug_gene"
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    model = BDRRAA(k = k, d = d, sample_size = 0.2, data = data, data2 = data2, non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem)
    iterations = 100
    model.train(iterations = iterations, LR = 0.1, print_loss = True)

    model.embedding_density(show = True)
    model.plot_auc()
    model.plot_latent_and_loss(iterations = iterations)
    model.plot_loss()
    # Plotting latent space
    Z_i = F.softmax(model.Z_i, dim=0)
    Z_j = F.softmax(model.Z_j, dim=0)
    Z = torch.cat((Z_i,Z_j),1)
    G = torch.sigmoid(model.Gate)
    C = (Z.T * G) / (Z.T * G).sum(0)


    embeddings = torch.matmul(model.A, torch.matmul(torch.matmul(Z, C), Z)).T
    #embeddings_j = torch.matmul(model.A_j, torch.matmul(torch.matmul(Z_j, C_j), Z_j)).T
    archetypes = torch.matmul(model.A, torch.matmul(Z, C))
    #archetypes_j = torch.matmul(model.A_j, torch.matmul(Z_j, C_j))


    fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2)
    sns.heatmap(Z.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax1)
    sns.heatmap(C.T.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax2)
    #sns.heatmap(Z_j.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax3)
    #sns.heatmap(C_j.T.detach().numpy(), cmap="YlGnBu", cbar=False, ax=ax4)

    if embeddings.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(embeddings[:, 0].detach().numpy(), embeddings[:, 1].detach().numpy(),
                   embeddings[:, 2].detach().numpy(), c='red')
        ax.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(),
                   archetypes[2, :].detach().numpy(), marker='^', c='black')
        ax.set_title(f"Latent space after {iterations} iterations")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.scatter(embeddings[model.sample_shape[0]:, 0].detach().numpy(), embeddings[model.sample_shape[0]:, 1].detach().numpy(), c='red')
        ax1.scatter(embeddings[:model.sample_shape[0], 0].detach().numpy(), embeddings[:model.sample_shape[0], 1].detach().numpy(), c='blue')
        ax1.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(), marker='^', c='black')
        ax1.set_title(f"Latent space after {iterations} iterations")
        # Plotting learning curve
        ax2.plot(model.losses)
        ax2.set_yscale("log")
        ax2.set_title("Loss")
    plt.show()

def main6():
    seed = 42
    torch.manual_seed(seed)
    k = 3
    d = 2

    # Data
    dataset = "cora"
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    model = KAAsparse(k = k, sample_size = 0.2, data = data, data2 = data2, non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem)
    iterations = 10000
    model.train(iterations = iterations, print_loss = True)
    auc, _, _ = model.link_prediction()
    print(auc)


def main7():
    seed = 42
    torch.manual_seed(seed)
    k = 3
    d = 2

    # Data
    dataset = "cora"
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    kaa = KAAsparse(k = k, sample_size = 0.2, data = data, data2 = data2, non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem)
    iterations = 10000
    kaa.train(iterations = 1000, print_loss = True)

    raa = DRRAA(data = data,
                    data_2 = data2,
                    k = k,
                    d = d,
                    data_type = "sparse",
                    sample_size=0.5,
                    link_pred = True,
                    non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem,
                    init_Z=kaa.S.detach()) # Set sampling procentage size
    raa.train(iterations = iterations, print_loss = True)
    auc, _, _ = raa.link_prediction()
    print(auc)

def main8():
    seed = 4
    torch.random.manual_seed(seed)
    # Data and hyperparameters
    dataset = "cora"
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    k = 3
    d = 2

    # Model
    model = LSM(data = data,
                    data_2 = data2,
                    d=d,
                    data_type = "sparse",
                    sample_size=0.2,
                    non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem) # Set sampling procentage size

    iterations = 2500
    model.train(iterations = iterations, LR = 0.1, print_loss = True)
    #model.plot_latent_and_loss(iterations = iterations) #TODO TypeError: float() argument must be a string or a number, not 'LinearSegmentedColormap'
    #model.embedding_density()
    model.plot_auc()
    #model.decision_boundary(attribute = node_attribute)


if __name__ == "__main__":
    main5()

from os import link
from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
import torch
import matplotlib.pyplot as plt
import numpy as np

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
    dataset = "astroph"
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    k = 5
    d = 2

    # Model
    model = DRRAA(data = data,
                    data_2 = data2,
                    k=k,
                    d=d,
                    data_type = "sparse",
                    sample_size=0.5,
                    non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem) # Set sampling procentage size

    iterations = 2500
    model.train(iterations = iterations, LR = 0.01, print_loss = True)
    model.plot_latent_and_loss(iterations = iterations)
    model.embedding_density()
    model.plot_auc()
    #model.decision_boundary(attribute = node_attribute)

def main4():
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

    d = 2
    k = 3

    # Model
    model = DRRAA(data = data,
                    data_2 = data2,
                    k = k,
                    d = d,
                    data_type = "sparse",
                    sample_size=0.5,
                    link_pred = True,
                    non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem) # Set sampling procentage size

    iterations = 25000
    model.train(iterations = iterations, LR = 0.1, print_loss = False, scheduling = True)
    model.plot_latent_and_loss(iterations = iterations)
    model.embedding_density()
    model.plot_auc()
    score, _, _ = model.link_prediction()
    print(score)

if __name__ == "__main__":
    main4()
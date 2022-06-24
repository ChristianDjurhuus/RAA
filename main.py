
from src.models.train_DRRAA_module import DRRAA
from src.models.train_DRRAA_nre import DRRAA_nre
from src.models.train_LSM_module import LSM, LSM_NRE
from src.models.train_BDRRAA_module import BDRRAA
from src.models.train_KAAsparse_module import KAAsparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import seaborn as sns

"""
This file demonstrates two different ways of using the RAA implementation. For other model implementations (like LDM)
inspect the models folder inside src.

The implementation can accommodate different data structures (see Preprocessing class for more information).
The data type is declared when initialising the class via the "data_type" variable.

"""

def main():
    seed = 42
    torch.random.manual_seed(seed)
    # Data and hyperparameters
    dataset = "facebook"
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    k = 3 # Number of archetypes
    d = 2 # Number of dimensions

    # Model
    model = DRRAA(data = data,
                    data_2 = data2,
                    k = k,
                    d = d,
                    data_type = "sparse",
                    sample_size=0.2,
                    link_pred = True,
                    non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem) # Set sampling procentage size

    iterations = 10000
    model.train(iterations = iterations, LR = 0.1, print_loss = True, scheduling = False)
    model.plot_latent_and_loss(iterations = iterations)
    model.embedding_density()
    model.plot_auc()
    score, _, _ = model.link_prediction()
    print(score)

def main2():
    seed = 42
    torch.random.manual_seed(seed)
    # Data and hyperparameters
    data = 'data/raw/polbooks/polbooks.gml'
    node_attribute = "value"

    k = 3 # Number of archetypes
    d = 2 # Number of dimensions

    # Model
    model = DRRAA(data = data,
                    k = k,
                    d = d,
                    data_type = "gml", 
                    sample_size = 1, link_pred = True) # Set sampling procentage size

    iterations = 2000
    model.train(iterations = iterations, LR = 0.01, print_loss = True)
    model.plot_latent_and_loss(iterations = iterations)
    model.embedding_density()
    model.plot_auc()

if __name__ == "__main__":
    main()
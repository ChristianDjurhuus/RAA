from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
import torch
import numpy as np

def get_embedding_density(datasets, sample_size, iterations, LR, print_loss = False):

    for idx, dataset in enumerate(datasets):
        print(dataset)
        # Load in data
        data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
        data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
        sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
        sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
        non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
        non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

        # model definition
        ldm = LSM(data = data,
                data_2 = data2,
                d = d,
                data_type = "sparse",
                sample_size = sample_size[idx],
                link_pred = True,
                non_sparse_i = non_sparse_i, non_sparse_j = non_sparse_j, sparse_i_rem = sparse_i_rem, sparse_j_rem = sparse_j_rem)

        ldm.train(iterations[idx], LR = LR, print_loss = print_loss)
        LDM_AUC, _, _ = ldm.link_prediction()
        print(f"{dataset}_auc: {LDM_AUC}")

        ldm.embedding_density(filename = f"{dataset}_embedding_density_LDM.pdf", show = False)

if __name__ == "__main__":
    datasets = ["cora", "facebook", "grqc", "hepth", "astroph", "dblp", "amazon", "youtube"]
    # Set iterations for each dataset
    iterations = [10000, 10000, 10000, 10000, 15000, 20000, 35000, 35000]
    # Set sample procentage for each dataset
    sample_size = [0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.05, 0.02]
    # Set if loss should be printed during training
    print_loss = False
    LR = 0.10

    # set dimensionality 
    d = 2
    ks = [3, 8]

    for l, dataset in enumerate(datasets):
        get_embedding_density([dataset], [sample_size[l]], [iterations[l]], LR, print_loss = print_loss)
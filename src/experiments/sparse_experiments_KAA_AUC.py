from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM, LSMAA
from src.models.train_KAAsparse_module import KAAsparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.stats as st
import matplotlib as mpl

def sparse_experiments(datasets, ks, sample_size, iterations, LR, print_loss = False):

    # Initialize datastructures for storing experiment data
    KAA_AUC_SCORES = {(key, k): [] for key in datasets for k in ks}

    for idx, dataset in enumerate(datasets):
        print(dataset)
        # Load in data
        data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
        data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
        sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
        sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
        non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
        non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

        for model_num in range(TRAIN_NUMS):
            print(f"Train_num iter: {model_num}")

            ## RAA
            # Cross validation loop
            for k in ks:
                print(f"kvals: {k}")
                # model definition
                kaa = KAAsparse(data = data,
                        data2 = data2,
                        k = k,
                        data_type = "sparse",
                        sample_size = sample_size[idx],
                        non_sparse_i = non_sparse_i, non_sparse_j = non_sparse_j, sparse_i_rem = sparse_i_rem, sparse_j_rem = sparse_j_rem)

                kaa.train(iterations[idx], LR = LR, print_loss = print_loss)

                KAA_AUC, _, _ = kaa.link_prediction()
                KAA_AUC_SCORES[dataset, k].append(KAA_AUC)

                kaa.embedding_density(filename = f"{dataset}_KAA_embedding_density_k{k}.pdf", show = False)

    # Create confidence interval for RAA
    lower_bound = [[] for d in datasets]
    upper_bound = [[] for d in datasets]
    for idx, dataset in enumerate(datasets):
        for k in ks:
            conf_interval = st.t.interval(alpha=0.95, df=len(KAA_AUC_SCORES[dataset, k]) - 1, loc=np.mean(KAA_AUC_SCORES[dataset, k]), scale=st.sem(KAA_AUC_SCORES[dataset, k]))
            lower_bound[idx].append(conf_interval[0])
            upper_bound[idx].append(conf_interval[1])

            
    with open(f"KAA_auc_{dataset}.txt", "w") as data:
        for k in ks:
            data.write(f"AUC for k: {k}")
            data.write("\n")
            data.write(json.dumps(KAA_AUC_SCORES[dataset, k]))
            data.write("\n")
            data.write(f"Mean AUC for RAA with {k} archetypes: {np.mean(KAA_AUC_SCORES[dataset, k])}")

        data.write("\n")
        data.write(f"confidence interval RAA lower and upper bound for each k:")
        data.write("\n")
        data.write(json.dumps(lower_bound))
        data.write(json.dumps(upper_bound))

if __name__ == "__main__":
    datasets = ["cora"]
    # Set iterations for each dataset
    iterations = [10000]
    # Set sample procentage for each dataset
    sample_size = [0.2]
    # Set if loss should be printed during training
    print_loss = False
    LR = 0.10

    TRAIN_NUMS = 5
    # Find seeds
    seeds = torch.randint(low = 0, high = 10000, size = (TRAIN_NUMS,))

    # set dimensionality 
    d = 2
    ks = [3, 8]

    for l, dataset in enumerate(datasets):
        sparse_experiments([dataset], ks, [sample_size[l]], [iterations[l]], LR, print_loss = True)
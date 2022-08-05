from src.models.train_DRRAA_nre import DRRAA_nre
from src.models.train_LSM_module import LSM_NRE
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.stats as st
import matplotlib as mpl

def sparse_experiments(datasets, ks, sample_size, iterations, LR, print_loss = False):

    # Initialize datastructures for storing experiment data
    LSM_AUC_SCORES = {key: [] for key in datasets}
    RAA_AUC_SCORES = {(key, k): [] for key in datasets for k in ks}

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

            # Initialize model
            lsm = LSM_NRE(data = data,
                data_2 = data2,
                d = d,
                data_type = "sparse",
                sample_size = sample_size[idx],
                link_pred = True,
                non_sparse_i = non_sparse_i, non_sparse_j = non_sparse_j, sparse_i_rem = sparse_i_rem, sparse_j_rem = sparse_j_rem)

            # Train lsm model
            lsm.train(iterations = iterations[idx], print_loss = print_loss)
            LSM_AUC_temp, _, _ = lsm.link_prediction()
            LSM_AUC_SCORES[dataset].append(LSM_AUC_temp)

            ## RAA
            # Cross validation loop
            for k in ks:
                print(f"kvals: {k}")
                # model definition
                raa = DRRAA_nre(data = data,
                        data_2 = data2,
                        k = k,
                        d = d,
                        data_type = "sparse",
                        sample_size = sample_size[idx],
                        link_pred = True,
                        non_sparse_i = non_sparse_i, non_sparse_j = non_sparse_j, sparse_i_rem = sparse_i_rem, sparse_j_rem = sparse_j_rem)

                raa.train(iterations[idx], LR = LR, print_loss = print_loss)
                RAA_AUC_TEMP, _, _ = raa.link_prediction()
                RAA_AUC_SCORES[dataset, k].append(RAA_AUC_TEMP)

    # Create confidence interval for RAA
    lower_bound = [[] for d in datasets]
    upper_bound = [[] for d in datasets]
    for idx, dataset in enumerate(datasets):
        for k in ks:
            conf_interval = st.t.interval(alpha=0.95, df=len(RAA_AUC_SCORES[dataset, k]) - 1, loc=np.mean(RAA_AUC_SCORES[dataset, k]), scale=st.sem(RAA_AUC_SCORES[dataset, k]))
            lower_bound[idx].append(conf_interval[0])
            upper_bound[idx].append(conf_interval[1])

    with open(f"noRE_auc_{dataset}.txt", "w") as data:
        data.write(f"{dataset}: AUC for LSM:")
        data.write("\n")
        data.write(json.dumps(LSM_AUC_SCORES[dataset]))
        data.write("\n")
        data.write(f"Mean AUC for LSM: {np.mean(LSM_AUC_SCORES[dataset])}")
        data.write("\n")
        for k in ks:
            data.write(f"AUC for k: {k}")
            data.write("\n")
            data.write(json.dumps(RAA_AUC_SCORES[dataset, k]))
            data.write("\n")
            data.write(f"Mean AUC for RAA with {k} archetypes: {np.mean(RAA_AUC_SCORES[dataset, k])}")
        data.write(f"confidence interval LSM:")
        data.write("\n")
        conf_interval_LSM = st.t.interval(alpha=0.95, df=len(LSM_AUC_SCORES[dataset]) - 1, loc=np.mean(LSM_AUC_SCORES[dataset]), scale=st.sem(LSM_AUC_SCORES[dataset]))
        data.write(json.dumps(conf_interval_LSM))
        data.write("\n")
        data.write(f"confidence interval RAA lower and uper bound for each k:")
        data.write("\n")
        data.write(json.dumps(lower_bound))
        data.write(json.dumps(upper_bound))

if __name__ == "__main__":
    datasets = ["cora"]
    # Set iterations for each dataset
    iterations = [15000]
    # Set sample procentage for each dataset
    sample_size = [1]
    # Set if loss should be printed during training
    print_loss = False
    LR = 0.010

    TRAIN_NUMS = 5
    # Find seeds
    seeds = torch.randint(low = 0, high = 10000, size = (TRAIN_NUMS,))

    # set dimensionality 
    d = 2
    ks = [3, 8]

    for l, dataset in enumerate(datasets):
        sparse_experiments([dataset], ks, [sample_size[l]], [iterations[l]], LR, print_loss = False)
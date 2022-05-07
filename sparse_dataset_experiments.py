from sympy import N
from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.stats as st
import matplotlib as mpl 
def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return
setup_mpl()

datasets = ["cora", "facebook"]
n_archetypes = torch.arange(2,12)
# Set iterations for each dataset
iterations = [15000, 15000]
# Set sample procentage for each dataset
sample_size = [1, 1]
# Set if loss should be printed during training
print_loss = False

# Initialize datastructures for storing experiment data
LSM_AUC_SCORES = {key: [] for key in datasets}
RAA_AUC_SCORES = {(key, k): [] for key in datasets for k in range(len(n_archetypes))}

TRAIN_NUMS = 5
# Find seeds
seeds = torch.randint(low = 0, high = 10000, size = (TRAIN_NUMS,))

# set dimensionality 
d = 2

for idx, dataset in enumerate(datasets):
    print(dataset)
    # Load in data
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

    # Training loop
    # Cross validation loop
    for model_num in range(TRAIN_NUMS):
        print(model_num)
        # Update seed
        torch.random.manual_seed(seeds[model_num])

        # Initialize model
        lsm = LSM(data = data,
            data_2 = data2,
            d = d,
            data_type = "sparse",
            sample_size = sample_size[idx],
            non_sparse_i = non_sparse_i, non_sparse_j = non_sparse_j, sparse_i_rem = sparse_i_rem, sparse_j_rem = sparse_j_rem)

        # Train lsm model
        lsm.train(iterations = iterations[idx], print_loss = print_loss)
        LSM_AUC, _, _ = lsm.link_prediction()
        # Append score
        LSM_AUC_SCORES[dataset].append(LSM_AUC)

        # Cross validation loop
        for k_vals in range(len(n_archetypes)):
            print(f"kvals: {k_vals}")
            # model definition
            raa = DRRAA(data = data,
                    data_2 = data2,
                    k = k_vals,
                    d = d,
                    data_type = "sparse",
                    sample_size = sample_size[idx],
                    non_sparse_i = non_sparse_i, non_sparse_j = non_sparse_j, sparse_i_rem = sparse_i_rem, sparse_j_rem = sparse_j_rem)

            raa.train(iterations[idx], print_loss = print_loss)
            RAA_AUC, _, _ = raa.link_prediction()
            # Append scores
            RAA_AUC_SCORES[dataset, k_vals].append(RAA_AUC)

# Plot the AUC scores with confidence intervals
# Create confidence interval for RAA
lower_bound = [[] for d in datasets]
upper_bound = [[] for d in datasets]
for idx, dataset in enumerate(datasets):
    for k_vals in range(len(n_archetypes)):
        conf_interval = st.t.interval(alpha=0.95, df=len(RAA_AUC_SCORES[dataset, k_vals]) - 1, loc=np.mean(RAA_AUC_SCORES[dataset, k_vals]), scale=st.sem(RAA_AUC_SCORES[dataset, k_vals]))
        lower_bound[idx].append(conf_interval[0])
        upper_bound[idx].append(conf_interval[1])


for idx, dataset in enumerate(datasets):
    fig, ax = plt.subplots(dpi = 100)

    # Calculate the confidence intervals
    conf_interval_LSM = st.t.interval(alpha=0.95, df=len(LSM_AUC_SCORES[dataset]) - 1, loc=np.mean(LSM_AUC_SCORES[dataset]), scale=st.sem(LSM_AUC_SCORES[dataset])) 
    #conf_interval_RAA = st.t.interval(alpha=0.95, df=len(RAA_AUC_SCORES[dataset]) - 1, loc=np.mean(RAA_AUC_SCORES[dataset]), scale=st.sem(RAA_AUC_SCORES[dataset]))

    LSM_line = [np.mean(LSM_AUC_SCORES[dataset]) for x in range(len(n_archetypes))]
    RAA_line = [np.mean(RAA_AUC_SCORES[dataset, k]) for k in range(len(n_archetypes))]

    # Plot AUC scores
    ax.plot(n_archetypes, LSM_line, label = dataset.title() + " LSM AUC", color = "blue")
    ax.plot(n_archetypes, RAA_line, label = dataset.title() + " RAA AUC", color = "orange")
    ax.plot(n_archetypes, LSM_line, marker = "o")
    ax.plot(n_archetypes, RAA_line, marker = "o")

    #plot confidence interval
    ax.fill_between(x = n_archetypes, y1 = [conf_interval_LSM[0] for i in range(len(n_archetypes))], y2 = [conf_interval_LSM[1] for i in range(len(n_archetypes))], color='tab:blue', alpha=0.2)
    ax.fill_between(x = n_archetypes, y1 = lower_bound[idx], y2 = upper_bound[idx], color='tab:orange', alpha=0.2)
    # set xlim
    ax.set_xlim(torch.min(n_archetypes), torch.max(n_archetypes))
    ax.set_xticks(n_archetypes)
    ax.set_ylabel("AUC score")
    ax.set_xlabel("k (number of archetypes)")

    ax.grid(alpha = 0.3)
    ax.legend()
    fig.savefig(dataset + "-AUC-score.pdf")
    #plt.show()

for idx, dataset in enumerate(datasets):
    fig, ax = plt.subplots(dpi = 100)

    # Calculate the confidence intervals
    conf_interval_LSM = st.t.interval(alpha=0.95, df=len(LSM_AUC_SCORES[dataset]) - 1, loc=np.mean(LSM_AUC_SCORES[dataset]), scale=st.sem(LSM_AUC_SCORES[dataset])) 
    #conf_interval_RAA = st.t.interval(alpha=0.95, df=len(RAA_AUC_SCORES[dataset]) - 1, loc=np.mean(RAA_AUC_SCORES[dataset]), scale=st.sem(RAA_AUC_SCORES[dataset]))

    LSM_line = [np.mean(LSM_AUC_SCORES[dataset]) for x in range(len(n_archetypes))]
    RAA_line = [np.mean(RAA_AUC_SCORES[dataset, k]) for k in range(len(n_archetypes))]

    # Plot AUC scores
    ax.plot(n_archetypes, LSM_line, label = dataset.title() + " LSM AUC", color = "blue")
    ax.plot(n_archetypes, RAA_line, label = dataset.title() + " RAA AUC", color = "orange")
    ax.plot(n_archetypes, LSM_line, marker = "o")
    ax.plot(n_archetypes, RAA_line, marker = "o")

    #plot confidence interval
    ax.fill_between(x = n_archetypes, y1 = [conf_interval_LSM[0] for i in range(len(n_archetypes))], y2 = [conf_interval_LSM[1] for i in range(len(n_archetypes))], color='tab:blue', alpha=0.2)
    ax.fill_between(x = n_archetypes, y1 = lower_bound[idx], y2 = upper_bound[idx], color='tab:orange', alpha=0.2)
    ax.plot(n_archetypes, [conf_interval_LSM[0] for i in range(len(n_archetypes))], linestyle = "--", color = "blue")
    ax.plot(n_archetypes, [conf_interval_LSM[1] for i in range(len(n_archetypes))], linestyle = "--", color = "blue")
    ax.plot(n_archetypes, lower_bound[idx], linestyle = "--", color = "orange")
    ax.plot(n_archetypes, upper_bound[idx], linestyle = "--", color = "orange")
    # set xlim
    ax.set_xlim(torch.min(n_archetypes), torch.max(n_archetypes))
    ax.set_xticks(n_archetypes)
    ax.set_ylabel("AUC score")
    ax.set_xlabel("k (number of archetypes)")

    ax.grid(alpha = 0.3)
    ax.legend()
    fig.savefig(dataset + "-AUC-score-withlines.pdf")
    #plt.show()

with open("sparse_experiment_cora_facebook.json", "w") as w:
    for dataset in datasets:
        data = json.dumps([LSM_AUC_SCORES[dataset] for x in range(len(n_archetypes))])
        w.write(data)
        data = json.dumps([RAA_AUC_SCORES[dataset, k] for k in range(len(n_archetypes))])
        w.write(data)
    data = json.dumps(lower_bound)
    w.write(data)
    data = json.dumps(upper_bound)
    w.write(data)
    data = json.dumps(conf_interval_LSM[0])
    w.write(data)
    data = json.dumps(conf_interval_LSM[1])
    w.write(data) 
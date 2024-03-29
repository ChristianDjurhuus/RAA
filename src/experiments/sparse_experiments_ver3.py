from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
from src.models.calcNMI import calcNMI
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.stats as st
import matplotlib as mpl

def sparse_experiments(datasets, sample_size, iterations, LR, n_archetypes, print_loss = False):

    # Initialize datastructures for storing experiment data
    LSM_AUC_SCORES = {key: [] for key in datasets}
    RAA_AUC_SCORES = {(key, k): [] for key in datasets for k in range(len(n_archetypes))}
    RAA_EMBEDDING = {(key, k): [] for key in datasets for k in range(len(n_archetypes))}
    RAA_NMI = {(key, k): [] for key in datasets for k in range(len(n_archetypes))}

    for idx, dataset in enumerate(datasets):
        print(dataset)
        # Load in data
        data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
        data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
        sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
        sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
        non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
        non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()
        seed = 0
        # Outer loop

        for model_num in range(TRAIN_NUMS):
            print(f"Train_num iter: {model_num}")
            # Update seed
            torch.random.manual_seed(seed)
            seed += 1

            # Initialize model
            lsm = LSM(data = data,
                data_2 = data2,
                d = d,
                data_type = "sparse",
                sample_size = sample_size[idx],
                link_pred = True,
                non_sparse_i = non_sparse_i, non_sparse_j = non_sparse_j, sparse_i_rem = sparse_i_rem, sparse_j_rem = sparse_j_rem)

            # Train lsm model
            lsm.train(iterations = iterations[idx], print_loss = print_loss)
            LSM_AUC, _, _ = lsm.link_prediction()
            LSM_AUC_SCORES[dataset].append(LSM_AUC)

            ## RAA
            # Cross validation loop
            for k_vals in range(len(n_archetypes)):
                k = n_archetypes[k_vals]
                print(f"kvals: {k_vals}")
                # model definition
                raa = DRRAA(data = data,
                        data_2 = data2,
                        k = k,
                        d = d,
                        data_type = "sparse",
                        sample_size = sample_size[idx],
                        link_pred = True,
                        non_sparse_i = non_sparse_i, non_sparse_j = non_sparse_j, sparse_i_rem = sparse_i_rem, sparse_j_rem = sparse_j_rem)

                raa.train(iterations[idx], LR = LR, print_loss = print_loss, scheduling = False)
                RAA_AUC_TEMP, _, _ = raa.link_prediction()
                RAA_AUC_SCORES[dataset, k_vals].append(RAA_AUC_TEMP)

                # Append embeddings
                Z = torch.softmax(raa.Z, dim=0)
                G = torch.sigmoid(raa.Gate)
                C = (Z.T * G) / (Z.T * G).sum(0)
                u, sigma, v = torch.svd(raa.A) # Decomposition of A.
                r = torch.matmul(torch.diag(sigma), v.T)
                embeddings = torch.matmul(r, torch.matmul(torch.matmul(Z, C), Z)).T
                embeddings = embeddings.cpu() #.detach().numpy()
                RAA_EMBEDDING[dataset, k_vals].append(embeddings)

    # Create confidence interval for RAA
    lower_bound = [[] for d in datasets]
    upper_bound = [[] for d in datasets]
    for idx, dataset in enumerate(datasets):
        for k_vals in range(len(n_archetypes)):
            conf_interval = st.t.interval(alpha=0.95, df=len(RAA_AUC_SCORES[dataset, k_vals]) - 1, loc=np.mean(RAA_AUC_SCORES[dataset, k_vals]), scale=st.sem(RAA_AUC_SCORES[dataset, k_vals]))
            lower_bound[idx].append(conf_interval[0])
            upper_bound[idx].append(conf_interval[1])


    # Plotting code
    for idx, dataset in enumerate(datasets):
        fig, ax = plt.subplots(dpi = 100)

        # Calculate the confidence intervals
        conf_interval_LSM = st.t.interval(alpha=0.95, df=len(LSM_AUC_SCORES[dataset]) - 1, loc=np.mean(LSM_AUC_SCORES[dataset]), scale=st.sem(LSM_AUC_SCORES[dataset])) 
        #conf_interval_RAA = st.t.interval(alpha=0.95, df=len(RAA_AUC_SCORES[dataset]) - 1, loc=np.mean(RAA_AUC_SCORES[dataset]), scale=st.sem(RAA_AUC_SCORES[dataset]))

        LSM_line = [np.mean(LSM_AUC_SCORES[dataset]) for x in range(len(n_archetypes))]
        RAA_line = [np.mean(RAA_AUC_SCORES[dataset, k]) for k in range(len(n_archetypes))]

        # Plot AUC scores
        ax.plot(n_archetypes, LSM_line, label = dataset.title() + " LSM AUC", color = "#e68653")
        ax.plot(n_archetypes, RAA_line, label = dataset.title() + " RAA AUC", color = "#e3427d")
        ax.plot(n_archetypes, LSM_line, marker = "o", color = "#e68653")
        ax.plot(n_archetypes, RAA_line, marker = "o", color = "#e3427d")

        #plot confidence interval
        ax.fill_between(x = n_archetypes, y1 = [conf_interval_LSM[0] for i in range(len(n_archetypes))], y2 = [conf_interval_LSM[1] for i in range(len(n_archetypes))], color="#e68653", alpha=0.2)
        ax.fill_between(x = n_archetypes, y1 = lower_bound[idx], y2 = upper_bound[idx], color="#e3427d", alpha=0.2)
        # set xlim
        ax.set_xlim(torch.min(n_archetypes), torch.max(n_archetypes))
        ax.set_xticks(n_archetypes)
        ax.set_ylabel("AUC score")
        ax.set_xlabel("k (number of archetypes)")

        ax.grid(alpha = 0.3)
        ax.legend()
        #fig.savefig(dataset + "-AUC-score.pdf")
        plt.show()

    for idx, dataset in enumerate(datasets):
        fig, ax = plt.subplots(dpi = 100)

        # Calculate the confidence intervals
        conf_interval_LSM = st.t.interval(alpha=0.95, df=len(LSM_AUC_SCORES[dataset]) - 1, loc=np.mean(LSM_AUC_SCORES[dataset]), scale=st.sem(LSM_AUC_SCORES[dataset])) 
        #conf_interval_RAA = st.t.interval(alpha=0.95, df=len(RAA_AUC_SCORES[dataset]) - 1, loc=np.mean(RAA_AUC_SCORES[dataset]), scale=st.sem(RAA_AUC_SCORES[dataset]))

        LSM_line = [np.mean(LSM_AUC_SCORES[dataset]) for x in range(len(n_archetypes))]
        RAA_line = [np.mean(RAA_AUC_SCORES[dataset, k]) for k in range(len(n_archetypes))]

        # Plot AUC scores
        ax.plot(n_archetypes, LSM_line, label = dataset.title() + " LSM AUC", color = "#e68653")
        ax.plot(n_archetypes, RAA_line, label = dataset.title() + " RAA AUC", color = "#e3427d")
        ax.plot(n_archetypes, LSM_line, marker = "o", color = "#e68653")
        ax.plot(n_archetypes, RAA_line, marker = "o", color = "#e3427d")

        #plot confidence interval
        ax.fill_between(x = n_archetypes, y1 = [conf_interval_LSM[0] for i in range(len(n_archetypes))], y2 = [conf_interval_LSM[1] for i in range(len(n_archetypes))], color = "#e68653", alpha=0.2)
        ax.fill_between(x = n_archetypes, y1 = lower_bound[idx], y2 = upper_bound[idx], color = "#e3427d", alpha = 0.2)
        ax.plot(n_archetypes, [conf_interval_LSM[0] for i in range(len(n_archetypes))], linestyle = "--", color = "#e68653")
        ax.plot(n_archetypes, [conf_interval_LSM[1] for i in range(len(n_archetypes))], linestyle = "--", color = "#e68653")
        ax.plot(n_archetypes, lower_bound[idx], linestyle = "--", color = "#e3427d")
        ax.plot(n_archetypes, upper_bound[idx], linestyle = "--", color = "#e3427d")
        # set xlim
        ax.set_xlim(torch.min(n_archetypes), torch.max(n_archetypes))
        ax.set_xticks(n_archetypes)
        ax.set_ylabel("AUC score")
        ax.set_xlabel("k (number of archetypes)")

        ax.grid(alpha = 0.3)
        ax.legend()
        #fig.savefig(dataset + "-AUC-score-withlines.pdf")
        plt.show() 
        # 
    # Compute NMIs
    
    NMIs = []
    for k in range(len(n_archetypes)):
        NMI = []
        idx = np.argmax(RAA_AUC_SCORES[dataset, k])
        for k_inner in range(TRAIN_NUMS):
            if k_inner != idx:
                NMI.append(float(calcNMI(RAA_EMBEDDING[dataset, k][idx].T, RAA_EMBEDDING[dataset, k][k_inner].T).detach().numpy()))
        NMIs.append(NMI)

    print(NMIs)

    with open(f"sparse_NMI_{dataset}.txt", "w") as data:
        data.write(json.dumps(NMIs))


    fig, ax = plt.subplots(figsize=(10,5), dpi=500)
    c='#e3427d'
    ax.boxplot(NMIs,patch_artist=True,
            boxprops=dict( facecolor=c,color=c),
            capprops=dict(color='#303638'),
            whiskerprops=dict(color='#303638'),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color='#303638'),
            )

    ax.set_ylabel("NMI")
    ax.set_xlabel("k: number of archetypes")
    ax.set_xlim(torch.min(n_archetypes), torch.max(n_archetypes))
    ax.set_xticks(n_archetypes)
    ax.grid(alpha=0.3)
    #plt.savefig(f"NMI_{dataset}.pdf", dpi=500)
    plt.show()

if __name__ == "__main__":
    datasets = ["cora"]
    n_archetypes = torch.arange(2,11)
    # Set iterations for each dataset
    iterations = [15000]
    # Set sample procentage for each dataset
    sample_size = [0.75]
    # Set if loss should be printed during training
    print_loss = False
    LR = 0.01

    TRAIN_NUMS = 3
    # Find seeds
    seeds = torch.randint(low = 0, high = 10000, size = (TRAIN_NUMS,))

    # set dimensionality 
    d = 2

    for l, dataset in enumerate(datasets):
        sparse_experiments([dataset], [sample_size[l]], [iterations[l]], LR, n_archetypes, print_loss = False)
        
"""
The goal of this file is to compare the performance of DRRAAA over a range
of number of archetypes and LSM. This comparison will be base uponlink prediction
and node classification (KNN and linear regression) performance over three dataset.
blogs, dolphin and karate.

"""
from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json

# Training iteration
iterations = 10000
d = 2 # Set dim for DRRAA
LSM_losses = []
n_archetypes = torch.arange(2,11)
# Find seeds
seeds = torch.randint(low = 0, high = 10000, size = (10,))
# Outer loop for the three dataset
dataset = ['data/raw/karate/karate.gml', 'data/raw/football/football.gml', 'data/raw/polblogs/polblogs.gml'] #, 'data/raw/dolphin/dolphins.gml'
data_names = ["Karate", "football","Polblogs"] # "Dolphins",
# TODO slight problem. Dolphins have no group metadata...
attribute_list = ["club", "value", "value"] # "label",

# Create dict for AA AUC score
AA_AUC_scores = {key: [] for key in data_names}
LSM_AUC_score_plot = {key: 0 for key in data_names}
LSM_KNN_plot = {key: 0 for key in data_names}
LSM_LR_plot = {key: 0 for key in data_names}
AA_KNN_plot = {key: [] for key in data_names}
AA_LR_plot = {key: [] for key in data_names}

for i, data in enumerate(dataset):
    print(data)
    # Train 10 LSM models and take the best
    LSM_AUC_score, LSM_KNN, LSM_LR = [], [], []
    for j, seed in enumerate(seeds): # Train model for each seed
        print(j)
        torch.random.manual_seed(seed)
        model = LSM(latent_dim = d,
                sample_size = 0.5,
                data = data,
                data_type = "gml")
        model.train(iterations = iterations, LR = 0.01, print_loss = False)
        LSM_AUC_score_temp, _, _ = model.link_prediction()
        LSM_AUC_score.append(LSM_AUC_score_temp)
        knn_score = model.KNeighborsClassifier(attribute = attribute_list[i])
        LSM_KNN.append(knn_score)
        lr_score = model.logistic_regression(attribute = attribute_list[i])
        LSM_LR.append(lr_score)
        LSM_losses.append(model.losses[-1]) #Append last loss of model i

    # Find the best LSM based on lowest loss
    min_LSM_loss = min(LSM_losses)
    min_LSM_loss_idx = LSM_losses.index(min_LSM_loss)
    LSM_AUC_score = [LSM_AUC_score[min_LSM_loss_idx] for x in range(len(n_archetypes))]
    LSM_KNN = [LSM_KNN[min_LSM_loss_idx] for x in range(len(n_archetypes))]
    LSM_LR = [LSM_LR[min_LSM_loss_idx] for x in range(len(n_archetypes))]
    # append to the plotting dict
    LSM_AUC_score_plot[data_names[i]] = LSM_AUC_score
    LSM_KNN_plot[data_names[i]] = LSM_KNN
    LSM_LR_plot[data_names[i]] = LSM_LR
    torch.random.manual_seed(seeds[min_LSM_loss_idx]) # Train AA with same seed

    for idx in range(len(n_archetypes)):
        print(f"arc_types: {idx}")
        model = DRRAA(data = data,
                        k=n_archetypes[idx],
                        d=d,
                        data_type = "gml", 
                        sample_size=0.5) # Set sampling procentage size
        model.train(iterations = iterations, LR = 0.01, print_loss = False)
        AA_AUC_score, _, _ = model.link_prediction()
        AA_AUC_scores[data_names[i]].append(AA_AUC_score)
        

        knn_score = model.KNeighborsClassifier(attribute = attribute_list[i])
        AA_KNN_plot[data_names[i]].append(knn_score)
        lr_score = model.logistic_regression(attribute = attribute_list[i])
        AA_LR_plot[data_names[i]].append(lr_score)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10))
for data_name in data_names:
    ax1.plot(n_archetypes, LSM_AUC_score_plot[data_name], label = data_name + " - LSM AUC")
    ax1.plot(n_archetypes, AA_AUC_scores[data_name], label = data_name + " - AA AUC")
ax1.set_xticks(n_archetypes)
ax1.set_xlim(torch.min(n_archetypes), torch.max(n_archetypes))
ax1.set_title(f"DRRAA vs. LSM with {iterations} iterations")
ax1.set_ylabel("AUC-score")
ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax1.grid(alpha = 0.3)

for data_name in data_names:
    ax2.plot(n_archetypes, LSM_KNN_plot[data_name], label = data_name + " - LSM KNN")
    ax2.plot(n_archetypes, LSM_LR_plot[data_name], label = data_name + " - LSM LR")
    ax2.plot(n_archetypes, AA_KNN_plot[data_name], label = data_name + " - AA KNN")
    ax2.plot(n_archetypes, AA_LR_plot[data_name], label = data_name + " - AA LR")
ax2.set_xticks(n_archetypes)
ax2.set_title("Node classification")
ax2.set_ylabel("Accuracy")
ax2.set_xlim(torch.min(n_archetypes), torch.max(n_archetypes))
ax2.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax2.grid(alpha = 0.3)
fig.tight_layout()
fig.savefig(fname = "alldata" + "_" + str(iterations) + "_iterations.png")
plt.show()


for data_name in data_names:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10))
    ax1.plot(n_archetypes, LSM_AUC_score_plot[data_name], label = data_name + " - LSM AUC")
    ax1.plot(n_archetypes, AA_AUC_scores[data_name], label = data_name + " - AA AUC")
    ax1.set_xticks(n_archetypes)
    ax1.set_xlim(torch.min(n_archetypes), torch.max(n_archetypes))
    ax1.set_title(f"DRRAA vs. LSM with {iterations} iterations")
    ax1.set_ylabel("AUC-score")
    ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax1.grid(alpha = 0.3)

    ax2.plot(n_archetypes, LSM_KNN_plot[data_name], label = data_name + " - LSM KNN")
    ax2.plot(n_archetypes, LSM_LR_plot[data_name], label = data_name + " - LSM LR")
    ax2.plot(n_archetypes, AA_KNN_plot[data_name], label = data_name + " - AA KNN")
    ax2.plot(n_archetypes, AA_LR_plot[data_name], label = data_name + " - AA LR")
    ax2.set_xticks(n_archetypes)
    ax2.set_title("Node classification")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlim(torch.min(n_archetypes), torch.max(n_archetypes))
    ax2.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax2.grid(alpha = 0.3)
    fig.tight_layout()
    fig.savefig(fname = data_name + "_" + str(iterations) + "_iterations.png")
    plt.show()

with open("comparison_data.json", "w") as w:
    data = json.dumps(LSM_AUC_score_plot)
    w.write(data)
    data = json.dumps(AA_AUC_scores)
    w.write(data)
    data = json.dumps(LSM_KNN_plot)
    w.write(data)
    data = json.dumps(LSM_LR_plot)
    w.write(data)
    data = json.dumps(AA_KNN_plot)
    w.write(data)
    data = json.dumps(AA_LR_plot)
    w.write(data) 
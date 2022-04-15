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

# Training iteration
iterations = 10
d = 2 # Set dim for DRRAA
LSM_losses = []
n_archetypes = torch.arange(2,11)
# Find seeds
seeds = torch.randint(low = 0, high = 10000, size = (10,))
# Outer loop for the three dataset
dataset = ['data/raw/karate/karate.gml', 'data/raw/dolphin/dolphins.gml''data/raw/polblogs/polblogs.gml']
data_names = ["karate", "dolphins", "polblogs"]
# TODO slight problem. Dolphins have no group metadata...
attribute_list = ["club", "label", "value"]

# Create dict for AA AUC score
AA_AUC_scores = {key: [] for key in data_names}
for i, data in tqdm(enumerate(dataset)):
    # Train 10 LSM models and take the best
    LSM_AUC_score, LSM_KNN, LSM_LR = [], [], []
    for seed in seeds: # Train model for each seed
        torch.random.manual_seed(seed)
        model = LSM(latent_dim = d,
                sample_size = 0.5,
                data = data,
                data_type = "gml")
        model.train(iterations = iterations, LR = 0.01, print_loss = False)
        LSM_AUC_score_temp, _, _ = model.link_prediction()
        knn_score = model.KNeighborsClassifier(attribute = attribute_list[i])
        LSM_KNN.append(knn_score)
        lr_score = model.logistic_regression(attribute = attribute_list[i])
        LSM_LR.append(lr_score)
        LSM_losses.append(model.losses[-1]) #Append last loss of model i

    # Find the best LSM based on lowest loss
    min_LSM_loss = min(LSM_losses)
    min_LSM_loss_idx = LSM_losses.index(min_LSM_loss)
    LSM_KNN = [LSM_KNN[min_LSM_loss_idx] for x in range(len(n_archetypes))]
    LSM_LR = [LSM_LR[min_LSM_loss_idx] for x in range(len(n_archetypes))]
    torch.random.manual_seed(seeds[min_LSM_loss_idx]) # Train AA with same seed

    AA_KNN = []
    AA_LR = []
    for idx in range(len(n_archetypes)):
        model = DRRAA(data = data,
                        k=n_archetypes[idx],
                        d=d,
                        data_type = "gml", 
                        sample_size=0.5) # Set sampling procentage size
        model.train(iterations = iterations, LR = 0.01, print_loss = False)
        AA_AUC_score, _, _ = model.link_prediction()
        AA_AUC_scores[data_names[i]].append(AA_AUC_score)
        

        knn_score = model.KNeighborsClassifier(attribute = attribute_list[i])
        AA_KNN.append(knn_score)
        lr_score = model.logistic_regression(attribute = attribute_list[i])
        AA_LR.append(lr_score)

LSM_AUC_scores = [LSM_AUC_score for x in range(len(n_archetypes))]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10), dpi=400)
ax1.plot(LSM_AUC_scores, label = "LSM AUC")
for data_name in data_names:
    ax1.plot(AA_AUC_scores[data_name], label = data_name)
ax1.set_xticks(n_archetypes.numpy())
ax1.set_title(f"DRRAA vs. LSM with {iterations} iterations")
ax1.set_ylabel("AUC-score")

ax2.plot(LSM_KNN, label = "LSM KNN")
ax2.plot(LSM_LR, label = "LSM LR")
ax2.plot(AA_KNN, label = "AA KNN")
ax2.plot(AA_LR, label = "AA LR")
ax2.set_xticks(n_archetypes.numpy())
ax2.set_title("Node classification")
ax2.set_ylabel("Score")
plt.show()
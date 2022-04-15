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

# Create dict for AA AUC score
AA_AUC_scores = {key: [] for key in data_names}
for i, data in tqdm(enumerate(dataset)):
    # Train 10 LSM models and take the best
    LSM_AUC_score = 0
    for seed in seeds: # Train model for each seed
        torch.random.manual_seed(seed)
        model = LSM(latent_dim = d,
                sample_size = 0.5,
                data = data,
                data_type = "gml")
        model.train(iterations = iterations, LR = 0.01, print_loss = False)
        LSM_AUC_score_temp, _, _ = model.link_prediction()
        if LSM_AUC_score_temp > LSM_AUC_score:
            LSM_AUC_score = LSM_AUC_score_temp
        LSM_losses.append(model.losses[-1]) #Append last loss of model i

    # Find the best LSM based on lowest loss
    min_LSM_loss = min(LSM_losses)
    min_LSM_loss_idx = LSM_losses.index(min_LSM_loss)
    torch.random.manual_seed(seeds[min_LSM_loss_idx]) # Train AA with same seed

    for idx in range(len(n_archetypes)):
        model = DRRAA(data = data,
                        k=n_archetypes[idx],
                        d=d,
                        data_type = "gml", 
                        sample_size=0.5) # Set sampling procentage size
        model.train(iterations = iterations, LR = 0.01, print_loss = False)
        AA_AUC_score, _, _ = model.link_prediction()
        AA_AUC_scores[data_names[i]].append(AA_AUC_score)

# Okay now we got a list for the best LSM_AUC for each dataset
# The AUC score for AA for each k \in{2:10}
# Time for some plotting (also need the classification)

LSM_AUC_scores = [LSM_AUC_score for x in range(len(n_archetypes))]

fig, ax = plt.subplots()
ax.plot(LSM_AUC_scores, label = "LSM AUC")
for data_name in data_names:
    ax.plot(AA_AUC_scores[data_name], label = data_name)
ax.set_xticks(n_archetypes.numpy())
ax.set_title(f"DRRAA vs. LSM with {iterations} iterations")
ax.set_ylabel("AUC-score")
plt.show()
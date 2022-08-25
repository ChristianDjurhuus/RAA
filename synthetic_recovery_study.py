import torch
import torch.nn.functional as F
import numpy as np
from src.models.train_DRRAA_module import DRRAA
from src.models.calcNMI import calcNMI


k = 3
d = 2
num_init = 5
iter = 1000
lr = 0.1

for alpha in [0.25, 1, 5]:
    dataset = f"synthetic{alpha}"
    data = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")).long()
    data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")).long()
    sparse_i_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
    sparse_j_rem = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
    non_sparse_i = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
    non_sparse_j = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")).long()
    true_Z = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/true_latent_Z.txt")).float()
    org_data =  torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_i.txt")).long()
    org_data2 = torch.from_numpy(np.loadtxt("data/train_masks/" + dataset + "/org_sparse_j.txt")).long()

    NMIS = np.zeros(num_init)
    seed = 0
    best_model = None
    for idx in range(num_init):
        best_loss = 1e5
        for _ in range(num_init):
            model = DRRAA(data = org_data,
                        data_2 = org_data2,
                        k = k,
                        d = d,
                        data_type = "sparse",
                        sample_size=1,
                        seed_init = seed,
                        link_pred = False,
                        non_sparse_i=non_sparse_i, non_sparse_j=non_sparse_j, sparse_i_rem=sparse_i_rem, sparse_j_rem=sparse_j_rem) # Set sampling procentage size

            model.train(iterations=iter, LR=lr)

            if np.mean(model.losses[-10:]) < best_loss:
                best_loss = np.mean(model.losses[-10:])
                nmi = calcNMI(model.latent_z.detach().cpu(), true_Z.T).item()
                best_model = model
                print(nmi, best_loss)
            seed += 1
        #print(nmi)
        NMIS[idx] = nmi
    print(np.mean(NMIS))
        #torch.save(best_model.state_dict(), f'synth_best_model_{nmi:.2f}_{alpha}.pt')
        #np.savetxt(f'Z_{nmi:.2f}_{alpha}', F.softmax(model.Z, 0).detach().numpy(),delimiter=',')
    #np.savetxt(f'nmis_{alpha}', NMIS, delimiter=',')

print(NMIS)
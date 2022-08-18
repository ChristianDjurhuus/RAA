import pandas as pd
import torch
from src.models.train_BDRRAA_module import BDRRAA
import networkx as nx
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.models.calcNMI import calcNMI
import matplotlib as mpl
from src.data.synthetic_data import truncate_colormap
import scipy

non_sparse_i = torch.from_numpy(np.loadtxt('data/train_masks/2020_congress/non_sparse_i.txt', delimiter=",")).long()
non_sparse_j = torch.from_numpy(np.loadtxt('data/train_masks/2020_congress/non_sparse_j.txt', delimiter=",")).long()
org_sparse_i = torch.from_numpy(np.loadtxt('data/train_masks/2020_congress/org_sparse_i.txt', delimiter=",")).long()
org_sparse_j = torch.from_numpy(np.loadtxt('data/train_masks/2020_congress/org_sparse_j.txt', delimiter=",")).long()
sparse_i = torch.from_numpy(np.loadtxt('data/train_masks/2020_congress/sparse_i.txt', delimiter=",")).long()
sparse_j = torch.from_numpy(np.loadtxt('data/train_masks/2020_congress/sparse_j.txt', delimiter=",")).long()
sparse_i_rem = torch.from_numpy(np.loadtxt('data/train_masks/2020_congress/sparse_i_rem.txt', delimiter=",")).long()
sparse_j_rem = torch.from_numpy(np.loadtxt('data/train_masks/2020_congress/sparse_j_rem.txt', delimiter=",")).long()

metadata = np.array(pd.read_csv('data/raw/2020_congress/2020_congress_metadata.csv', index_col=False))[:,1:]
missing_data = np.array(pd.read_csv('data/raw/2020_congress/2020_congress_missing_data_idx.csv', index_col=False))[:,1:]



model = BDRRAA(k=3,
                    d=2,
                    sample_size=0.5,
                    data=sparse_i,
                    data2=sparse_j, non_sparse_i=non_sparse_i,non_sparse_j=non_sparse_j,sparse_i_rem=sparse_i_rem,sparse_j_rem=sparse_j_rem,
                    data_type = "sparse"
        )
model.train(iterations=10, print_loss=True)




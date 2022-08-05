import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.models.train_DRRAA_module import DRRAA
import torch
import numpy as np

def show_embeddings(datasets, ks, sample_size, iterations, LR, print_loss = False):
    for idx, dataset in enumerate(datasets):
        print(dataset)
        # Load in data
        data = torch.from_numpy(np.loadtxt("/zhome/a4/3/146946/Desktop/s194245/Bachelor/RAA/data/train_masks/" + dataset + "/sparse_i.txt")).long()
        data2 = torch.from_numpy(np.loadtxt("/zhome/a4/3/146946/Desktop/s194245/Bachelor/RAA/data/train_masks/" + dataset + "/sparse_j.txt")).long()
        sparse_i_rem = torch.from_numpy(np.loadtxt("/zhome/a4/3/146946/Desktop/s194245/Bachelor/RAA/data/train_masks/" + dataset + "/sparse_i_rem.txt")).long()
        sparse_j_rem = torch.from_numpy(np.loadtxt("/zhome/a4/3/146946/Desktop/s194245/Bachelor/RAA/data/train_masks/" + dataset + "/sparse_j_rem.txt")).long()
        non_sparse_i = torch.from_numpy(np.loadtxt("/zhome/a4/3/146946/Desktop/s194245/Bachelor/RAA/data/train_masks/" + dataset + "/non_sparse_i.txt")).long()
        non_sparse_j = torch.from_numpy(np.loadtxt("/zhome/a4/3/146946/Desktop/s194245/Bachelor/RAA/data/train_masks/" + dataset + "/non_sparse_j.txt")).long()

        # Create full graph for visualisation
        i_partion = np.concatenate((data.cpu(), sparse_i_rem.cpu()))
        j_partion = np.concatenate((data2.cpu(), sparse_j_rem.cpu()))
        edge_list = np.zeros((2, len(i_partion)))
        for idx in range(len(i_partion)):
            edge_list[0, idx] = i_partion[idx]
            edge_list[1, idx] = j_partion[idx]
        edge_list = list(zip(edge_list[0], edge_list[1]))
        # Make graph
        G = nx.from_edgelist(edge_list)

        for k in ks:
            raa = DRRAA(d=2, k=k, data=G, data_type='networkx',link_pred = False, sample_size=sample_size[0])

            raa.train(iterations[0], LR = LR, print_loss = print_loss, scheduling = False)
            #get colorlist
            #color_list = ["303638","f0c808","5d4b20","469374","9341b3","e3427d","e68653","ebe0b0","edfbba","ffadad","ffd6a5","fdffb6","caffbf","9bf6ff","a0c4ff","bdb2ff","ffc6ff","fffffc"]
            #color_list = ["#"+i.lower() for i in color_list]
            #color_map = [color_list[14] if G.nodes[i]['value'] == 0 else color_list[5] for i in G.nodes()]
            color_map = "#e68653"
            #draw graph
            d = dict(G.degree)
            embeddings, archetypes = raa.get_embeddings()
            pos_map = dict(list(zip(G.nodes(), list(embeddings))))
            nx_pos_map = nx.spring_layout(G)
            fig, ax = plt.subplots(dpi=200)
            nx.draw_networkx_nodes(G, pos=nx_pos_map, ax = ax, node_color=color_map, alpha=.9, node_size=[v for v in d.values()])
            nx.draw_networkx_edges(G, pos=nx_pos_map, ax = ax, alpha=.2)
            fig.savefig(f"spring_{dataset}_k{k}.pdf", dpi = 200)

            fig, ax = plt.subplots(dpi=200)
            nx.draw_networkx_nodes(G, pos=pos_map, ax=ax, node_color=color_map, alpha=.9, node_size=[v for v in d.values()])
            nx.draw_networkx_edges(G, pos=pos_map, ax=ax, alpha=.1)
            ax.scatter(archetypes[0, :], archetypes[1, :], marker='^', c='black', label="Archetypes", s=80)
            ax.legend()
            fig.savefig(f"show_embedding_{dataset}_k{k}.pdf", dpi = 200)
            ax.cla()

            raa.embedding_density(filename = f"show_embedding_density_{dataset}_k{k}.pdf", show = False)


if __name__ == "__main__":
    datasets = ["cora", "facebook", "grqc", "hepth", "astroph"]
    # Set iterations for each dataset
    iterations = [15000, 15000, 15000, 20000, 20000]
    # Set sample procentage for each dataset
    sample_size = [1, 1, 1, 1, 1]
    # Set if loss should be printed during training
    print_loss = False
    LR = 0.010

    # set dimensionality 
    d = 2
    ks = [3, 8]

    for l, dataset in enumerate(datasets):
        show_embeddings([dataset], ks, [sample_size[l]], [iterations[l]], LR, print_loss = print_loss)
    

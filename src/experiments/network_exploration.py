import networkx as nx
import community as community_louvain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import netwulf as nw
from collections import defaultdict

# Load in data
datasets = ["cora", "facebook"]

def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = plt.figure(figsize=(5, 5), dpi = 200) # in inches
    plt.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")
    
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = plt.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)
    fig.savefig(f"adjacency_{dataset}.pdf")
    plt.show()

for dataset in datasets:
    sparse_i = np.loadtxt("data/train_masks/" + dataset + "/sparse_i.txt")
    sparse_j = np.loadtxt("data/train_masks/" + dataset + "/sparse_j.txt")
    sparse_i_rem = np.loadtxt("data/train_masks/" + dataset + "/sparse_i_rem.txt")
    sparse_j_rem = np.loadtxt("data/train_masks/" + dataset + "/sparse_j_rem.txt")
    non_sparse_i = np.loadtxt("data/train_masks/" + dataset + "/non_sparse_i.txt")
    non_sparse_j = np.loadtxt("data/train_masks/" + dataset + "/non_sparse_j.txt")


    # Collect the entire graph
    i_partion = np.concatenate((sparse_i, sparse_i_rem))
    j_partion = np.concatenate((sparse_j, sparse_j_rem))
    edge_list = np.zeros((2, len(i_partion)))
    for idx in range(len(i_partion)):
        edge_list[0, idx] = i_partion[idx]
        edge_list[1, idx] = j_partion[idx]
    edge_list = list(zip(edge_list[0], edge_list[1]))
    
    # Make graph
    G = nx.from_edgelist(edge_list)
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    print(f"Number of nodes in {dataset}: {G.number_of_nodes()}")
    print(f"Number of edges in {dataset}: {G.number_of_edges()}")

    num_components = nx.number_connected_components(G)
    print(f"The number of connected components for {dataset.title()} is: {num_components}")

    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, dpi = 200)
    # Find the adjacency matrix for plotting
    adjacency_matrix = nx.to_numpy_matrix(G)
    print(adjacency_matrix.shape)
    print(f"len: {np.sum(adjacency_matrix)}")
    #ax1.imshow(adjacency_matrix, cmap="Greys",
    #              interpolation="none") # , cmap = "BrBG"
    #plt.show()

    # Find communities based upon the Louvain community algorithm
    #communities = community_louvain.best_partition(G)
    #for k, node in G.nodes(data=True):
    #    node['detected_group'] = communities[k]
    #    node['size'] = G.degree[k] # Set size to degree

    #with plt.style.context('ggplot'):
    #    network, config = nw.visualize(nw.get_filtered_network(G, node_group_key='detected_group'), plot_in_cell_below=False)
    #    fig, ax = nw.draw_netwulf(network)
    #    fig.savefig(dataset + "netwulf.pdf", dpi = 1000)

    #fig, ax = plt.subplots(dpi = 100)
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(dataset + "network.pdf")
    #plt.show()

    # Run louvain community finding algorithm
    #louvain_community_dict = community_louvain.best_partition(G)

    # Convert community assignmet dict into list of communities
    #louvain_comms = defaultdict(list)
    #for node_index, comm_id in louvain_community_dict.items():
    #    louvain_comms[comm_id].append(node_index)
    #louvain_comms = louvain_comms.values()

    #nodes_louvain_ordered = [node for comm in louvain_comms for node in comm]
    draw_adjacency_matrix(G)
    #draw_adjacency_matrix(G, nodes_louvain_ordered, [louvain_comms], ["blue"])

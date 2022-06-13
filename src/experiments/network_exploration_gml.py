import networkx as nx
import community as community_louvain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import netwulf as nw
from collections import defaultdict
import torch
import matplotlib.patches as mpatches

# Load in data
dataset = "polblogs"

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


#import data
G = nx.read_gml("data/raw/polblogs/polblogs.gml")
G = G.to_undirected()


if nx.number_connected_components(G) > 1:
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
label_map = {x: i for i, x in enumerate(G.nodes)}
G = nx.relabel_nodes(G, label_map)

adj_m = nx.adjacency_matrix(G)
temp_G = nx.from_numpy_matrix(adj_m)
#nx.set_node_attributes(temp_G, nx.get_node_attributes(G,"value"))
temp_G.graph.update({"value":0})
for u, v in temp_G.nodes(data=True):
    temp_G.nodes[u]["value"] = nx.get_node_attributes(G, "value")[u]

#get colorlist
color_list = ["303638","f0c808","5d4b20","469374","9341b3","e3427d","e68653","ebe0b0","edfbba","ffadad","ffd6a5","fdffb6","caffbf","9bf6ff","a0c4ff","bdb2ff","ffc6ff","fffffc"]
color_list = ["#"+i.lower() for i in color_list]
#color_map = [color_list[14] if G.nodes[i]['value'] == 0 else color_list[5] for i in G.nodes()]
topic_list = np.unique(list(nx.get_node_attributes(temp_G, "value").values()))
color_map = dict(zip(topic_list, [color_list[14],color_list[5]]))

degree_sequence = sorted((d for n, d in temp_G.degree()), reverse=True)
dmax = max(degree_sequence)

print(f"Number of nodes in {dataset}: {temp_G.number_of_nodes()}")
print(f"Number of edges in {dataset}: {temp_G.number_of_edges()}")

num_components = nx.number_connected_components(temp_G)
print(f"The number of connected components for {dataset.title()} is: {num_components}")

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, dpi = 200)
# Find the adjacency matrix for plotting
adjacency_matrix = nx.to_numpy_matrix(temp_G)
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

for k, v in temp_G.nodes(data = True):
    v["group"] = v["value"]; del v["value"]
    v['size'] = temp_G.degree[k]

patches = []
with plt.style.context('ggplot'):
    network, config = nw.interactive.visualize(temp_G, plot_in_cell_below=False)
    for i in range(len(network['nodes'])):
        network['nodes'][i]['color'] = color_map[nx.get_node_attributes(temp_G,'group')[network['nodes'][i]['id']]]
    fig, ax = nw.draw_netwulf(network)
    #for (t,c) in color_map.items():
    #    patches.append(mpatches.Patch(color=c, label=t))
    #lgd = plt.legend(handles=patches,bbox_to_anchor=(1.04,1), loc="upper left")
    #text = plt.text(-0.2,1.05, " ", transform=ax.transAxes)
    plt.savefig(dataset + "netwulf.pdf", bbox_inches='tight', dpi=500) #,bbox_extra_artists=(lgd,text)

#with plt.style.context('ggplot'):
#    network, config = nw.visualize(nw.get_filtered_network(temp_G, node_group_key='group'), plot_in_cell_below=False)
#    fig, ax = nw.draw_netwulf(network)
#    fig.savefig(dataset + "netwulf.pdf", dpi = 500)

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
draw_adjacency_matrix(temp_G)
#draw_adjacency_matrix(G, nodes_louvain_ordered, [louvain_comms], ["blue"])

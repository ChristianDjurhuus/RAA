import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#graph:
np.random.seed(42)
G = nx.Graph()
fig = plt.figure(figsize=(3,3),dpi=200)
G.add_nodes_from([1,2,3,4,5,6])
G.add_edges_from([(1,2),(2,3),(4,5),(6,5),(3,4),(4,6),(3,1)])
nx.draw(G, node_color='#e3427d')
G2 = nx.Graph()
G2.add_nodes_from([1,2])
nx.draw(G2, node_color='#5d4b20')
plt.savefig('Graph0.png',dpi=200)
#plt.show()

#graph with removed component:
np.random.seed(42)
G = nx.Graph()
fig = plt.figure(figsize=(3,3),dpi=200)
G.add_nodes_from([1,2,3,4,5,6])
G.add_edges_from([(1,2),(2,3),(4,5),(6,5),(3,4),(4,6),(3,1)])
nx.draw(G, node_color='#e3427d')
plt.savefig('Graph1.png',dpi=200)
#plt.show()

#graph with removed edge:
np.random.seed(42)
G = nx.Graph()
fig = plt.figure(figsize=(3,3),dpi=200)
G.add_nodes_from([1,2,3,4,5,6])
G.add_edges_from([(1,2),(2,3), (4,5),(6,5),(3,4),(4,6),(3,1)])
edge_style = ['-','-','-','-','-','--','-']
nx.draw(G, style=edge_style,node_color ='#e3427d')
plt.savefig('Graph2.png',dpi=200)
#plt.show()

#graph with removed edge disconnecting components
np.random.seed(42)
G = nx.Graph()
fig = plt.figure(figsize=(3,3),dpi=200)
G.add_nodes_from([1,2,3,4,5,6])
G.add_edges_from([(1,2),(2,3), (4,5),(6,5),(3,4),(4,6),(3,1)])
edge_style = ['-','-','-','--','-','-','-']
nx.draw(G, style=edge_style,node_color ='#e3427d')
plt.savefig('Graph3.png',dpi=200)
#plt.show()
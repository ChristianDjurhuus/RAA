from py_pcha import PCHA
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np

ZKC_graph = nx.karate_club_graph()
#Let's keep track of which nodes represent John A and Mr Hi
Mr_Hi = 0
John_A = 33

#Let's display the labels of which club each member ended up joining
club_labels = nx.get_node_attributes(ZKC_graph,'club')

#Getting adjacency matrix
A = nx.convert_matrix.to_numpy_matrix(ZKC_graph)

K = A.T@A 
XC, S, C, SSE, varexpl = PCHA(K, noc=2)

sigma = np.std(A)
K = np.exp(-((A**2)/sigma))
#K = data @ data.T
XC, S, C, SSE, varexpl = PCHA(K, noc=2)
_, new_archetypal_coords, C, _, _ = PCHA(K, noc=2)
new_archetypes = np.array(A.T @ C).T
new_archetypal_coords = np.array(new_archetypal_coords.T)

embeddings = np.asarray(XC.T)
archetypes = np.asarray(np.dot(S, C))

labels = list(club_labels.values())
idx_hi = [i for i, x in enumerate(labels) if x == "Mr. Hi"]
idx_of = [i for i, x in enumerate(labels) if x == "Officer"]

if embeddings.shape[1] == 3:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embeddings[:,0].detach().numpy()[idx_hi], embeddings[:,1].detach().numpy()[idx_hi], embeddings[:,2].detach().numpy()[idx_hi], c = 'red', label='Mr. Hi' )
    ax.scatter(embeddings[:,0].detach().numpy()[idx_of], embeddings[:,1].detach().numpy()[idx_of], embeddings[:,2][idx_of].detach().numpy(), c = 'blue', label='Officer')
    ax.scatter(archetypes[0,:].detach().numpy(), archetypes[1,:].detach().numpy(), archetypes[2,:].detach().numpy(), marker = '^', c='black')
    ax.text(embeddings[Mr_Hi,0].detach().numpy(), embeddings[Mr_Hi,1].detach().numpy(), embeddings[Mr_Hi,2].detach().numpy(), 'Mr. Hi')
    ax.text(embeddings[John_A, 0].detach().numpy(), embeddings[John_A, 1].detach().numpy(), embeddings[John_A, 2].detach().numpy(),  'Officer')
    ax.set_title(f"Latent space after {iterations} iterations")
    ax.legend()
else:
    fig, ax1 = plt.subplots()
    ax1.scatter(embeddings[0,:][idx_hi], embeddings[1,:][idx_hi], c = 'red', label='Mr. Hi')
    ax1.scatter(embeddings[0,:][idx_of], embeddings[1,:][idx_of], c = 'blue', label='Officer')
    ax1.scatter(archetypes[0,:], archetypes[0,:], marker = '^', c = 'black')
    ax1.annotate('Mr. Hi', embeddings[:,Mr_Hi])
    ax1.annotate('Officer', embeddings[:,John_A])
    ax1.legend()
    ax1.set_title("KAA")
plt.show()
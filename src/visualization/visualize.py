import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import numpy as np
from fast_histogram import histogram2d

class Visualization():
    def __init__(self) -> None:
        pass

    def get_embeddings(self):
        Z = torch.softmax(self.Z, dim=0)
        G = torch.sigmoid(self.G)
        C = (Z.T * G) / (Z.T * G).sum(0)

        embeddings = torch.matmul(self.A, torch.matmul(torch.matmul(Z, C), Z)).T
        embeddings = embeddings.cpu()
        archetypes = torch.matmul(self.A, torch.matmul(Z, C))
        archetypes = archetypes.cpu()
        return embeddings, archetypes


    def plot_latent_and_loss(self, iterations):
        embeddings, archetypes = self.get_embeddings()
        if embeddings.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(embeddings[:, 0].detach().numpy(), embeddings[:, 1].detach().numpy(),
                    embeddings[:, 2].detach().numpy(), c='red')
            ax.scatter(archetypes[0, :].detach().numpy(), self.archetypes[1, :].detach().numpy(),
                    self.archetypes[2, :].detach().numpy(), marker='^', c='black')
            ax.set_title(f"Latent space after {iterations} iterations")
            #ax.legend()
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.scatter(embeddings[:, 0].detach().numpy(), embeddings[:, 1].detach().numpy(), c='red')
            ax1.scatter(archetypes[0, :].detach().numpy(), archetypes[1, :].detach().numpy(), marker='^', c='black')
            ax1.legend()
            ax1.set_title(f"Latent space after {iterations} iterations")
            # Plotting learning curve
            ax2.plot(self.losses)
            ax2.set_yscale('log') 
            ax2.set_title("Loss")
        plt.show()


    def embedding_density(self):
        embeddings, archetypes = self.get_embeddings()
        bounds = [[embeddings.detach().numpy()[:, 0].min(), embeddings.detach().numpy()[:, 0].max()], [embeddings.detach().numpy()[:, 1].min(), embeddings.detach().numpy()[:, 1].max()]]
        #print(bounds)
        #h = histogram2d(embeddings.detach().numpy()[:, 0], embeddings.detach().numpy()[:, 1], range = bounds, bins = round(np.sqrt(self.N)))
        x_bins = np.linspace(embeddings.detach().numpy()[:, 0].min(), embeddings.detach().numpy()[:, 0].max(), round(np.sqrt(self.N)))
        y_bins = np.linspace(embeddings.detach().numpy()[:, 1].min(), embeddings.detach().numpy()[:, 1].max(), round(np.sqrt(self.N)))
        plt.hist2d(embeddings.detach().numpy()[:, 0], embeddings.detach().numpy()[:, 1], cmap = "OrRd", bins = [x_bins, y_bins], norm=colors.LogNorm())
        #plt.plot(h, cmap = "OrRd", norm=colors.LogNorm())
        #plt.axis('off')
        plt.colorbar()
        plt.show()
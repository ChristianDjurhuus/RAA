import torch
import torch.nn as nn
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import Softmax
import torch.nn.functional as F


class RAA(nn.Module):
    def __init__(self, A, input_size, k):
        super(RAA, self).__init__()  # What the heck?!?
        self.A = A
        self.input_size = input_size
        self.k = k

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size[1]))
        self.a = torch.nn.Parameter(torch.randn(1))
        self.Z = torch.nn.Parameter(
            torch.randn(self.k, self.input_size[0])
        )  # Should probably be changed to a single N
        self.C = torch.nn.Parameter(torch.randn(self.input_size[0], self.k))
        self.latent_zi = torch.nn.Parameter(torch.randn(self.input_size[0], self.k))
        self.latent_zj = torch.nn.Parameter(torch.randn(self.input_size[1], self.k))

        self.softmax = Softmax(dim=1)

    def log_likelihood(self):
        # TODO
        # don't sum over i==j
        # Constraints are not implemented sum(Z) = 1 and Z > 0 same goes for C
        z_dist = (
            (
                (
                    torch.unsqueeze(
                        torch.matmul(
                            self.latent_zi,
                            torch.matmul(
                                F.softmax(self.Z, dim=1), F.softmax(self.C, dim=1)
                            ),
                        ),
                        1,
                    )
                    - torch.matmul(
                        self.latent_zj,
                        torch.matmul(
                            F.softmax(self.Z, dim=1), F.softmax(self.C, dim=1)
                        ),
                    )
                    + 1e-06
                )
                ** 2
            ).sum(-1)
        ) ** 0.5

        bias_matrix = torch.unsqueeze(self.beta, 1) + self.gamma
        theta = bias_matrix - self.a * z_dist
        LL = ((theta) * self.A).sum() - torch.sum(torch.log(1 + torch.exp(theta)))
        return LL


if __name__ == "__main__":
    seed = 1998
    torch.random.manual_seed(seed)

    A = mmread("data/raw/soc-karate.mtx")
    A = A.todense()
    A = torch.from_numpy(A)
    k = 2

    model = RAA(A=A, input_size=A.shape, k=k)
    optimizer = torch.optim.Adam(params=model.parameters())

    losses = []
    iterations = 50000
    for _ in range(iterations):
        loss = -model.log_likelihood() / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print("Loss at the", _, "iteration:", loss.item())

    def setup_mpl():
        mpl.rcParams["font.family"] = "Helvetica Neue"

    setup_mpl()

    # Plotting latent space
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=400)
    latent_zi = model.latent_zi.data.numpy()
    latent_zj = model.latent_zj.data.numpy()
    ax1.scatter(latent_zi[:, 0], latent_zj[:, 1])
    ax1.set_title(f"Latent space after {iterations} iterations")
    # Plotting learning curve
    ax2.plot(losses)
    ax2.set_title("Loss")
    plt.show()

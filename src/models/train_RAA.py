from re import L, S
import torch
import torch.nn as nn

class RAA(nn.Module):
    def __init__(self, A, input_size, k):
        self.A = A
        self.input_size = input_size
        self.k = k

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size[1]))
        self.a = torch.nn.Parameter(torch.randn(1))
        self.Z = torch.nn.parameter(torch.randn(self.k, self.input_size[0])) #Should probably be changed to a single N
        self.C = torch.nn.parameter(torch.randn(self.input_size[0], self.k)) #Should probably be changed to a single N
        self.latent_zi = torch.nn.Parameter(torch.randn(self.input_size[0], self.k))
        self.latent_zj = torch.nn.Parameter(torch.randn(self.input_size[1], self.k))

    def log_likelihood(self):
        #TODO don't sum over i==j
        z_dist = (((torch.unsqueeze(torch.matmul(torch.matmul(self.Z, self.C), self.latent_zi), 1) - torch.matmul(torch.matmul(self.Z, self.C), self.latent_zj) + 1e-06 )**2 ).sum(-1))**0.5
        bias_matrix = torch.unsqueeze(self.beta, 1) + self.gamma
        theta = bias_matrix - self.a * z_dist

        LL = ((theta) * self.A).sum() - torch.sum(torch.log(1 + torch.exp(theta)))

        return LL

if __name__ == "__main__":
    










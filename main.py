
from src.models.train_DRRAA_module import DRRAA
import torch
import matplotlib.pyplot as plt

def main():
    seed = 4
    torch.random.manual_seed(seed)
    # Data and hyperparameters
    data = 'data/raw/polblogs/polblogs.gml'
    k = 10
    d = 10

    # Model
    model = DRRAA(data = data,
                    k=k,
                    d=d,
                    data_type = "gml", 
                    sample_size=0.5) # Set sampling procentage size

    # Train
    iterations = 10000
    model.train(iterations = iterations, LR = 0.01, print_loss = True)
    # Visualization
    model.plot_latent_and_loss(iterations)
    model.embedding_density()

    # Link prediction
    model.plot_auc()
    
if __name__ == "__main__":
    main()
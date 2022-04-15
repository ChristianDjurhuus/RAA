
from src.models.train_DRRAA_module import DRRAA
from src.models.train_LSM_module import LSM
import torch
import matplotlib.pyplot as plt

def main():
    seed = 4
    torch.random.manual_seed(seed)
    # Data and hyperparameters
    data = 'data/raw/karate/karate.gml'
    node_attribute = "club"
    k = 10
    d = 2

    # Model
    model = DRRAA(data = data,
                    k=k,
                    d=d,
                    data_type = "gml", 
                    sample_size=0.5) # Set sampling procentage size

    #model = LSM(latent_dim = d,
    #                sample_size = 0.5,
    #                data = data,
    #                data_type = "gml")
    # Train
    iterations = 4500
    model.train(iterations = iterations, LR = 0.01, print_loss = True)
    # Visualization
    model.plot_latent_and_loss(iterations)
    model.embedding_density()
    knn_score = model.KNeighborsClassifier(attribute = node_attribute)
    print(f"knn_score: {knn_score}")
    log_reg = model.logistic_regression(attribute = node_attribute)
    print(f"logistic regression score: {log_reg}")
    # Link prediction
    model.plot_auc()
    
if __name__ == "__main__":
    main()
from tqdm import tqdm
import toml
import torch
from torch.utils.data import DataLoader

from nodes.CustomDataset import CustomDataset
from nodes.DiabedNet import DiabedNet
from nodes.TrainingNN import TrainingNN
from utils_local.ClassificationMetrics import ClassificationMetrics
from utils_local.utils import random_seed, plotting, summary_model

def main() -> None:
    config = toml.load("configs/config.toml")
    random_seed()

    # Dataset
    train_dataset = CustomDataset(config, train=True)
    test_dataset = CustomDataset(config, train=False)

    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Model
    model = DiabedNet(config) 
                    
    # Training procedure
    training_NN = TrainingNN(config, train_dataloader, test_dataloader)
    training_NN.training(model)
    train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history = training_NN.get_history()

    # Results test    
    if config['results']['metrics']:
        classification_metrics = ClassificationMetrics(test_dataloader, model)
        # classification_metrics.report()
        print(f'\nClassification Report:\n {classification_metrics.report()}')
    
    if config['results']['summary_net']: 
        print(summary_model(model, config))
    
    if config['results']['plotting_loss']: 
        plotting(train_loss_history, test_loss_history, "Loss")    
    
    if config['results']['plotting_acc']:
        plotting(train_accuracy_history, test_accuracy_history, "Accuracy")

if __name__ == "__main__":
    main()

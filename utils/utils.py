import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary

def plotting(train_history, test_history, title):
    plt.plot(train_history, label='train')
    plt.plot(test_history, label='test')
    plt.legend(loc='best')
    plt.title(title)

def plotting_pr(precision, recall, ap):
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'AP = {ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()




# Fixed seed
def random_seed():    
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    torch.cuda.manual_seed(9)
    torch.backends.cudnn.deterministic = True

def summary_model(model, config): 
    input_size = (1, config["model"]["input_size"])
    return summary(model, input_size=input_size)

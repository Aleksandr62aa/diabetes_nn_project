import torch


class DiabedNet(torch.nn.Module):
    def __init__(self, config):
        # super().__init__()
        super(DiabedNet, self).__init__()
        input_size = config["model"]["input_size"]
        n_hidden_1 = config["model"]["n_hidden_1"]
        n_hidden_2 = config["model"]["n_hidden_2"]
        n_hidden_3 = config["model"]["n_hidden_3"]
        classes = config["model"]["classes"]
        self.fc1 = torch.nn.Linear(input_size, n_hidden_1)
        self.fc2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = torch.nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = torch.nn.Linear(n_hidden_3, classes)
        self.relu = torch.nn.ReLU()
        # self.drop_out = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
import torch
from sklearn.metrics import accuracy_score, classification_report

class ClassificationMetrics():
    def __init__(self, test_dataloader):
        self.test_dataloader = test_dataloader
        self.y_true = []
        self.y_pred = []

    def report(self, model):        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device) 
        for batch in self.test_dataloader:
            X_batch, y_batch = batch
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, y_preds = torch.max(outputs, dim=1)
            self.y_pred.extend(y_preds.cpu().detach().numpy())
            self.y_true.extend(y_batch.cpu().detach().numpy())
        return classification_report(self.y_true, self.y_pred)
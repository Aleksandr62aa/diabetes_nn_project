import torch
from sklearn.metrics import accuracy_score, classification_report, average_precision_score, precision_recall_curve

class ClassificationMetrics():
    def __init__(self, test_dataloader, model):
        self.model = model
        self.test_dataloader = test_dataloader
        self.y_true = []
        self.y_pred = []
        self.test()
    
    def test(self):        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device) 
        for batch in self.test_dataloader:
            X_batch, y_batch = batch
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, y_preds = torch.max(outputs, dim=1)
            self.y_pred.extend(y_preds.cpu().detach().numpy())
            self.y_true.extend(y_batch.cpu().detach().numpy())    
    
    def report(self):      
        return classification_report(self.y_true, self.y_pred)
        
    def average_precision(self):      
        return average_precision_score(self.y_true, self.y_pred)
    
    def precision_recall(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred)      
        return precision, recall

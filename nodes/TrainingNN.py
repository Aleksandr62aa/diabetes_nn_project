import torch
from tqdm import tqdm

class TrainingNN():
    def __init__(self, config, train_dataloader, test_dataloader):
        self.lr = config["training"]["learning_rate"]
        self.num_epochs = config["training"]["num_epochs"]
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader        
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []        

    def training(self, model, loss=None, optimizer=None):
        if loss is None:
            loss = torch.nn.CrossEntropyLoss(torch.tensor([0.3, 0.7]))
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device) 
        for epoch in tqdm(range(self.num_epochs)):   # цикл обучения по эпохам
            running_loss_train = 0.
            running_acc_train = 0.
            model.train()

            for batch in self.train_dataloader:# цикл обучения по бачам
                optimizer.zero_grad()
                X_batch, y_batch = batch

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_preds = model.forward(X_batch)

                # loss функция
                loss_train = loss(y_preds, y_batch)

                loss_train.backward()
                optimizer.step()
                running_loss_train += loss_train.cpu().detach().numpy()
                running_acc_train += (y_preds.argmax(dim=1) == y_batch).float().mean().cpu().detach().numpy()
            epoch_loss = running_loss_train/ len(self.train_dataloader)
            self.train_loss_history.append(epoch_loss)

            epoch_acc = running_acc_train/len(self.train_dataloader)
            self.train_accuracy_history.append(epoch_acc)

            # тестирование модели
            running_loss_test = 0.
            running_acc_test = 0.
            model.eval()

            for batch in self.test_dataloader:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_preds = model.forward(X_batch)

                running_loss_test += loss(y_preds, y_batch).cpu().detach().numpy()
                running_acc_test += (y_preds.argmax(dim=1) == y_batch).float().mean().cpu().detach().numpy()

            epoch_loss = running_loss_test/len(self.test_dataloader)
            self.test_loss_history.append(epoch_loss)

            epoch_acc = running_acc_test/len(self.test_dataloader)
            self.test_accuracy_history.append(epoch_acc)

    def get_history(self):
        return self.train_loss_history, self.train_accuracy_history, self.test_loss_history, self.test_accuracy_history
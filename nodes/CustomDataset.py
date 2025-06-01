import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, config, train=True, train_ratio=0.8):
        """
        Class, представляющий набор данных.

        Attributes:
            tile_name (str): имя файла Dataset.
            train (bool): обучающая выборка.
            train_ratio (float): относительный размер обущающей выборки.
        """
        self.file_data = config["dataset"]["file_data"]        
        self.train = train
        self.train_ratio = train_ratio
        self.load_data()

    def load_data(self) -> None:
        """
        Read csv file

        Returns:
            None
        """       
        data_frame = pd.read_csv(self.file_data)

        # Разделить данные на признаки и целевую переменную
        x_np = data_frame.iloc[:,:-1].values
        y_np = data_frame.iloc[:,-1].values

        # Applying StandardScaler
        scaler = StandardScaler()
        x_np_tran = scaler.fit_transform(x_np)

        # Разделить данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
                                    x_np_tran, y_np, train_size=self.train_ratio, shuffle=False)

        if self.train:
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.int64)

    def __len__(self) -> int:
        """
        Возвращает количество элементов в наборе данных.

        Returns:
            int: Количество элементов в наборе данных.
        """
        return len(self.y)

    def __getitem__(self, index: int) -> tuple:
        """
        Возвращает элемент набора данных по указанному индексу.

        Args:
            index (int): Индекс элемента.
        Returns:
            tuple: Кортеж, содержащий элемент набора данных и соответствующий ему класс.
        """
        return self.X[index], self.y[index]
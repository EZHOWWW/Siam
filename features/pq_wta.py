# Некачественное ГДИС
# Англ.: Poor Quality Well Test Analysis
# Сокращение: PQ_WTA
from base_feature import BaseFeature
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from oildataset import (
    Series,
    BinFeatures,
    ValueFeatures,
    OilDataset,
    oil_dataset,
    oil_dataloader,
    oil_train_dataloader,
    oil_test_dataloader,
)
import pickle


class PQ_WTA(BaseFeature):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()

    def extract(self, s: Series):
        features = []
        for col in [1, 2]:
            i = s[col].numpy()
            diff = np.diff(i)
            features += [
                np.std(i),  # Шум
                np.percentile(np.abs(diff), 95),  # Скачки
            ]
        return np.array(features, dtype=np.float64)

    def prepire(self, X):
        return [self.extract(x) for x in X]

    def train(self, X, y_bin, y_val=None) -> None:
        self.model.fit(self.prepire(X), np.array(y_bin))

    def train_epoch(self, loader, lr: float = 0.03, device: str = "cuda") -> None:
        print("start")
        for (_, X), (y_bin, _) in loader:
            y_bin = y_bin[:, 0]
            self.train(X, y_bin)

    def predict(self, series: Series) -> tuple[BinFeatures, ValueFeatures]:
        return self.predict_bin(series), "No"

    def predict_bin(self, series: Series):
        return self.model.predict(self.prepire(series))

    def loss(self, Y_true, Y_pred):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(torch.tensor(Y_true), torch.tensor(Y_pred))

    def loss_data(self, dataloader):
        Y_true, Y_pred = [], []
        for (_, X), (y_true, _) in dataloader:
            y_true = y_true[:, 0]
            Y_true += [*y_true]
            Y_pred += [torch.tensor(*self.predict_bin(X))]
        return self.loss(np.array(Y_true), np.array(Y_pred))


pqwta = PQ_WTA()
(a, b), (c, d) = oil_dataset[0]
print(b, c)
pqwta.train_epoch(oil_train_dataloader)
pickle.dump(pqwta.model, open("pq_wtabodel.sav", "wb"))
print(pqwta.loss_data(oil_test_dataloader))

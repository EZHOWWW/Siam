from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from numpy import nan

from oildataset import (
    Series,
    BinFeatures,
    ValueFeatures,
    OilDataset,
    oil_dataloader,
    oil_dataset,
)


class BaseFeature(ABC):
    def __init__(self, nun_value=nan):
        self.nun_value = nun_value

    @abstractmethod
    def predict(self, series: Series) -> tuple[BinFeatures, ValueFeatures]:
        """
        Сделать предикт. Получаеть серию и выдает tuple(bin_feature, details)

        RETURN: tuple(
                Бинарный признак, есть фича или нет,
                Числовой признак, если фичи нет то self.nun_value)
        """

    ...

    @abstractmethod
    def train(self, X: Series, y_bin: BinFeatures, y_val: ValueFeatures) -> None:
        """
        Обучить модель.

        X: Контейнер из серий.
        y_bin: Контейнер такого же размера из бинарных признаков
        y_val: Конейнер такого же размера из числовых признаков
        """

    ...

    @abstractmethod
    def train_epoch(
        self, loader: DataLoader, lr: float = 0.03, device: str = "cuda"
    ) -> None: ...

    @abstractmethod
    def loss(Y_true, Y_pred) -> tuple[float, float]:
        """
        Получить значение ошибки, ошибка для бинарного признака, ошибка для числового
        """

    ...

from abc import ABC, abstractmethod
from numpy import nan


class BaseFeature(ABC):
    def __init__(self, nun_value=nan):
        self.nun_value = nun_value

    @abstractmethod
    def predict(self, series) -> tuple[float, float]:
        """
        Сделать предикт. Получаеть серию и выдает tuple(bin_feature, details)

        RETURN: tuple(
                Бинарный признак, есть фича или нет,
                Числовой признак, если фичи нет то self.nun_value)
        """
    ...
    
    @abstractmethod
    def train(X, y_bin, y_val) -> None:
        """
        Обучить модель. 

        X: Контейнер из серий.
        y_bin: Контейнер такого же размера из бинарных признаков
        y_val: Конейнер такого же размера из числовых признаков
        """

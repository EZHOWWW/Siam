import fireducks.pandas as pd
import typing as tp
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import PchipInterpolator
from sklearn.preprocessing import StandardScaler, RobustScaler


Series = tp.Union[torch.tensor, None]
BinFeatures = torch.tensor
ValueFeatures = torch.tensor


class OilDataset(Dataset):
    def __init__(self, path_to_data_folder="../Data"):
        super().__init__()
        self.path_to_data_folder = path_to_data_folder
        self.data = pd.concat(self.read_data(path_to_data_folder))
        self.bin_features = [
            "Некачественное ГДИС",
            "Влияние ствола скважины",
            "Радиальный режим",
            "Линейный режим",
            "Билинейный режим",
            "Сферический режим",
            "Граница постоянного давления",
            "Граница непроницаемый разлом",
        ]
        self.value_features = [f + "_details" for f in self.bin_features[1:]]
        self._series_size = 512

    def read_data(self, path_to_data_folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        hq_file_name = "hq_markup_train.csv"
        lq_file_name = "markup_train.csv"
        hq_file_path = os.path.join(path_to_data_folder, hq_file_name)
        lq_file_path = os.path.join(path_to_data_folder, lq_file_name)

        return pd.read_csv(hq_file_path), pd.read_csv(lq_file_path)

    def _series_trainsform(self, s):
        # Интерполяция с сохранением временной оси
        if len(s.index) < 2:
            return torch.zeros(3, self._series_size)
        time_original = s["Time"].values
        time_new = np.linspace(
            time_original.min(), time_original.max(), self._series_size
        )

        # Нормализация времени
        time_scaled = (time_new - time_new.min()) / (
            time_new.max() - time_new.min() + 1e-8
        )

        # Интерполяция остальных каналов
        delta_p_interp = PchipInterpolator(time_original, s["DeltaP"])(time_new)
        p_prime_interp = PchipInterpolator(time_original, s["P_prime"])(time_new)

        # Нормализация данных
        data = np.vstack(
            [
                time_scaled,
                StandardScaler().fit_transform(delta_p_interp.reshape(-1, 1)).flatten(),
                StandardScaler().fit_transform(p_prime_interp.reshape(-1, 1)).flatten(),
            ]
        )
        return torch.tensor(data, dtype=torch.float32)  # [3, 512]

    def _data_transform(self, loc):
        # Бинарные признаки
        bin_data = torch.tensor(loc[self.bin_features].values.astype(np.float32))  # [8]

        # Числовые детали с учетом порядка режимов
        details = loc[self.value_features].copy()

        # Замена NaN и масштабирование с сохранением порядка признаков
        details_filled = details.fillna(-1).values
        # details_filled = details.values
        details_scaled = RobustScaler().fit_transform(
            details_filled.reshape(-1, 1)
        )  # [7]

        return (bin_data, torch.tensor(details_scaled, dtype=torch.float32))

    def _get_series_from_loc(self, loc):
        file_path = os.path.join(self.path_to_data_folder, "data", loc["file_name"])
        try:
            df = pd.read_csv(
                file_path,
                sep="\t",
                header=None,
                names=["Time", "DeltaP", "P_prime"],
                dtype={"Time": "float32", "DeltaP": "float32", "P_prime": "float32"},
                engine="c",
            )
            return df._evaluate()
        except FileNotFoundError as x:
            print(file_path)
            print(f"File not found in path:\t {file_path}\n" + x)
            return None

    def transform(self, s, d):
        return (s, self._series_trainsform(s)), self._data_transform(d)

    def __len__(self) -> int:
        return len(self.data.index)

    def __getitem__(self, idx: int) -> tuple[Series, tuple[BinFeatures, ValueFeatures]]:
        loc = self.data.iloc[idx]
        return self.transform(self._get_series_from_loc(loc), loc)


BATCH_SIZE = 512
oil_dataset = OilDataset()
oil_dataset_train, oil_dataset_test = torch.utils.data.random_split(
    oil_dataset, [0.9, 0.1]
)
oil_train_dataloader = DataLoader(oil_dataset_train, batch_size=BATCH_SIZE)
oil_test_dataloader = DataLoader(oil_dataset_test, batch_size=BATCH_SIZE)
oil_dataloader = DataLoader(oil_dataset, batch_size=BATCH_SIZE)

oil_head_dataset = OilDataset()
oil_head_dataset.data = oil_head_dataset.data.head(512)
oil_train_head_dataloader = DataLoader(oil_head_dataset, batch_size=BATCH_SIZE)

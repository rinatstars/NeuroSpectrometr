import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict

class SpectraDictDataset(Dataset):
    """
    PyTorch Dataset для списка словарей с данными спектров.
    Формат словаря (entry):
    {
        "sample": str,
        "parallel": int,
        "time": float,
        "label": int,              # метка для классификации
        "spectrum": np.ndarray,    # спектр после маски/преобразования
        "raw_spectrum": np.ndarray,# исходный спектр
        "wavelengths": np.ndarray,
        "raw_wavelengths": np.ndarray,
        "intensity": float         # интенсивность (регрессия)
    }
    """
    def __init__(self, data: List[Dict], transform: Optional[callable] = None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        spectrum = entry["spectrum"]
        raw_spectrum = entry["raw_spectrum"]
        label = entry["label"]
        intensity = entry["intensity"]

        # При желании применяем transform к spectrum
        if self.transform:
            spectrum = self.transform(spectrum)
            raw_spectrum = self.transform(raw_spectrum)

        # Преобразуем в тензоры
        spectrum = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)       # (1, seq_len)
        raw_spectrum = torch.tensor(raw_spectrum, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)                            # для классификации
        intensity = torch.tensor(intensity, dtype=torch.float32)                 # для регрессии

        return {
            "spectrum": spectrum,
            "raw_spectrum": raw_spectrum,
            "label": label,
            "intensity": intensity
        }

def load_dataset_from_npy(npy_file: str) -> List[Dict]:
    """
    Загружает список словарей из .npy файла.

    Parameters:
    - npy_file: путь к файлу .npy с сохранённым списком словарей.

    Returns:
    - list с записями (entry).
    """
    try:
        loaded = np.load(npy_file, allow_pickle=True)
        data_list = loaded.tolist()
        print(f"Загружено {len(data_list)} записей из {npy_file}")
        return data_list
    except Exception as e:
        print(f"Ошибка загрузки данных из {npy_file}: {e}")
        return []

def prepare_training_data(dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Извлекает и формирует numpy массивы для обучения из списка словарей.
    Возвращает:
    - spectra_masked (np.ndarray): массив спектров после маскировки/преобразования (N, seq_len)
    - spectra_raw (np.ndarray): исходные спектры (N, seq_len)
    - class_targets (np.ndarray): метки для классификации (N,)
    - reg_targets (np.ndarray): значения интенсивности (N,)

    Parameters:
    - dataset: список записей (entry)

    Возвращаемые значения:
    - spectra_masked, spectra_raw, class_targets, reg_targets
    """
    spectra_masked = []
    spectra_raw = []
    class_targets = []
    reg_targets = []

    for entry in dataset:
        spectra_masked.append(entry["spectrum"])
        spectra_raw.append(entry["raw_spectrum"])
        class_targets.append(entry["label"])
        reg_targets.append(entry["intensity"])

    # Преобразуем в numpy массивы
    spectra_masked = np.array(spectra_masked, dtype=np.float32)
    spectra_raw = np.array(spectra_raw, dtype=np.float32)
    class_targets = np.array(class_targets, dtype=np.int64)
    reg_targets = np.array(reg_targets, dtype=np.float32)

    return spectra_masked, spectra_raw, class_targets, reg_targets


def split_dataset(
    dataset: List[Dict],
    val_ratio: float = 0.2,
    shuffle: bool = True,
    seed: Optional[int] = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Делит единый список записей dataset на обучающую и валидационную части.

    Parameters:
    - dataset: полный список словарей с данными
    - val_ratio: доля данных для валидации (по умолчанию 20%)
    - shuffle: перемешивать ли данные перед разделением (рекомендуется True)
    - seed: сид для воспроизводимости рандома

    Returns:
    - train_dataset: список записей для обучения
    - val_dataset: список записей для валидации
    """
    if seed is not None:
        random.seed(seed)

    data = dataset.copy()

    if shuffle:
        random.shuffle(data)

    n_val = int(len(data) * val_ratio)
    val_dataset = data[:n_val]
    train_dataset = data[n_val:]

    print(f"Данные разделены: train = {len(train_dataset)} записей, val = {len(val_dataset)} записей")

    return train_dataset, val_dataset
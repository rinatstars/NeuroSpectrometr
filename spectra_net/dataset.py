import torch
from torch.utils.data import Dataset
import numpy as np

class SpectraDataset(Dataset):
    def __init__(self, data_list, transform=None, device='cpu'):
        """
        data_list: список словарей с полями
          - 'masked_spectrum' (нормализованный спектр фиксированной длины)
          - 'raw_spectrum' (исходный спектр)
          - 'wavelengths' (соответствующая длина волн, фиксированной длины)
          - 'label' (метка класса)
          - 'intensity' (регрессионная цель)
        """
        self.data_list = data_list
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]

        spectrum = torch.tensor(entry['spectrum'], dtype=torch.float32, device=self.device)
        wavelengths = torch.tensor(entry['wavelengths'], dtype=torch.float32, device=self.device)
        raw_spectrum = spectrum

        # Применение трасформера к спектру
        if self.transform:
            spectrum = self.transform(spectrum)
            spectrum = torch.tensor(spectrum, dtype=torch.float32, device=self.device)

        # Применяем маску к нормализованному спектру
        x_with_mask = add_wavelength_mask(spectrum, wavelengths)
        raw_spectrum = add_wavelength_mask(raw_spectrum, wavelengths)

        label = torch.tensor(int(entry['label']), dtype=torch.long, device=self.device)
        intensity = torch.tensor(entry['intensity'], dtype=torch.float32, device=self.device)

        return {
            'masked_spectrum': x_with_mask,   # [2, длина спектра]
            'raw_spectrum': raw_spectrum,     # [размерность raw]
            'label': label,
            'intensity': intensity
        }


class SpectrumTransform:
    """
    Класс преобразований (нормализация, шум и т.п.)
    """

    def __init__(self, method='minmax', noise_std=0.0):
        self.method = method
        self.noise_std = noise_std

    def __call__(self, spectrum):
        spectrum = np.array(spectrum, dtype=np.float32)
        if self.method == 'minmax':
            spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
        elif self.method == 'zscore':
            spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)
        if self.noise_std > 0:
            spectrum += np.random.normal(0, self.noise_std, size=spectrum.shape)
        return spectrum

def add_wavelength_mask(spectrum: torch.Tensor, wavelengths: torch.Tensor,
                        target_wavelength: float = 267.595, tol: float = 0.05) -> torch.Tensor:
    """
    Добавляет маску для спектра с учетом целевой длины волны.
    Возвращает тензор размера [2, длина спектра]:
    [интенсивность, маска].
    """
    diod = wavelengths[1] - wavelengths[0]
    idx = torch.argmin(torch.abs(wavelengths - target_wavelength))

    delta = (wavelengths[idx] - target_wavelength) / diod

    mask = torch.zeros_like(spectrum)
    if idx > 0:
        mask[idx - 1] = 0.25 + delta / 2
    mask[idx] = 0.5
    if idx < len(mask) - 1:
        mask[idx + 1] = 0.25 - delta / 2

    return torch.stack([spectrum, mask], dim=0)
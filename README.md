# NeuroSpectrometr

NeuroSpectrometr — это высокоэффективный фреймворк для анализа спектральных данных, использующий сверточно-внимательные нейронные сети (ConvAttentionNet) для одновременной классификации спектров и регрессии интенсивности их пиков. Проект ориентирован на исследовательские и производственные задачи в оптико-электронной области.

---

## Основные возможности

- Модель с комбинированным выходом: бинарная классификация + регрессия интенсивности.
- Современная архитектура с Conv и MultiheadAttention.
- Гибкая система трансформаций спектров (нормализация, добавление шума).
- Настраиваемые функции потерь с балансировкой компонентов.
- Легко расширяемый и масштабируемый код с модульной структурой.
- Полный pipeline: загрузка данных, обучение, валидация, сохранение модели.
- Поддержка TensorBoard для визуализации метрик обучения.

---

## Структура проекта
```bash
spectra_project/
│
├── spectra_net/ # Основной пакет с реализацией
│ ├── __init__.py
│ ├── model.py # Модель ConvAttentionNet
│ ├── loss.py # Функции потерь
│ ├── dataset.py # Dataset и трансформации
| ├── dataloader.py # Dataloader с подготовкой данных на обучение и валидацию
│ ├── trainer.py # Класс Trainer
│ ├── utils.py # Утилиты (загрузка конфига, сохранение модели и др.)
│
├── configs/
│ └── config.yaml # Конфигурация параметров обучения и модели
│
├── main.py # Главный скрипт запуска обучения
├── requirements.txt # Зависимости
└── README.md # Документация
```
---

## Быстрый старт

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Подготовьте данные в формате .npy, соответствующие структуре:
```bash
entry = {
         "sample": sample_name,               # Имя образца
         "parallel": parallel_id,             # Номер параллельного измерения
         "time": time,                        # Время спектра в выгораниях
         "label": label,                      # Ожидаемый ответ нейросети
         "spectrum": spectrum,                # Интерполированный спектр
         "raw_spectrum": orig_spectra[idx],   # Оригинальный спектр
         "wavelengths": wavelengths,          # Длины волн интерпол. спектра
         "raw_wavelengths": orig_wavelengths, # Оригинальные длины волн
         "intensity": intensities[idx]        # Интенсивность спектральной линии
     }
```
3. Настройте конфигурацию configs/config.yaml

4. Запустите обучение:

```bash
python main.py
```
5. Для мониторинга обучения используйте TensorBoard:
```bash
tensorboard --logdir runs/
```

##Конфигурация
В файле configs/config.yaml задаются параметры модели, пути к данным, параметры обучения и трансформаций. Пример:
```bash
data:
  dataset_npy: "D:/Data/neuro/dataset_labeled.npy"

training:
  batch_size: 64
  epochs: 60
  learning_rate: 0.001
  reg_weight: 1
  save_dir: "checkpoints/"

model:
  input_channels: 2
  conv_channels: [64, 32]
  kernel_size: [5, 3]
  attn_heads: 4
  dropout: 0.1849
  num_classes: 2

transform:
  method: "minmax"
  noise_std: 0.01

loss:
  class_weight: 1.0
  reg_weight: 1.0
```
## Архитектура

- spectra_net/model.py — модель с Conv и MultiheadAttention слоями.
- spectra_net/dataloader.py — загрузка спектров, подготовка train и val выборок
- spectra_net/dataset.py —  создание датасета с нормализацией и аугментацией для нейросети.
- spectra_net/loss.py — комбинированная функция потерь для классификации и регрессии.
- spectra_net/trainer.py — цикл обучения и валидации с логированием.
- spectra_net/utils.py — вспомогательные функции.

## Контакты
Если у вас возникли вопросы или предложения, пишите:

Email: tolyadzyuba@vmk.ru

GitHub: https://github.com/rinatstart/NeuroSpectrometr

## Лицензия
MIT License © 2025
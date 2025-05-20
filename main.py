import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from spectra_net.model import ConvAttentionNet
from spectra_net.loss import CombinedLoss
from spectra_net.dataset import SpectraDataset, SpectrumTransform
from spectra_net.trainer import Trainer
from spectra_net.utils import load_config
from spectra_net.dataloader import load_dataset_from_npy, split_dataset

def main():
    # Загружаем конфиг
    config = load_config('configs/config.yaml')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")

    # Загружаем полный датасет
    dataset = load_dataset_from_npy(config['data']['dataset_npy'])

    transform = SpectrumTransform(method=config['transform']['method'],
                                  noise_std=config['transform']['noise_std'])

    # Делим на train и val
    train_data, val_data = split_dataset(dataset, val_ratio=0.2)

    # Создаём объекты Dataset
    train_dataset = SpectraDataset(train_data, transform=transform, device=device)
    val_dataset = SpectraDataset(val_data, transform=transform, device=device)

    # Создаём DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Создаем модель
    model = ConvAttentionNet(
        num_classes=config['model']['num_classes'],
        input_channels=config['model']['input_channels'],
        conv_channels1=config['model']['conv_channels'][0],
        conv_channels2=config['model']['conv_channels'][1],
        kernel_size1=config['model']['kernel_size'][0],
        kernel_size2=config['model']['kernel_size'][1],
        attn_heads=config['model']['attn_heads'],
        dropout=config['model']['dropout'],
    )

    model.to(device)

    # Создаем функцию потерь
    criterion = CombinedLoss(class_weight=config['loss']['class_weight'],
                             reg_weight=config['loss']['reg_weight'])

    # Оптимизатор
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # TensorBoard writer (опционально)
    writer = SummaryWriter(log_dir='runs/spectra_experiment')

    # Создаем тренер
    trainer = Trainer(model=model,
                      dataloaders=dataloaders,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      save_dir=config['training']['save_dir'],
                      writer=writer,
                      reg_weight=config['training']['reg_weight'])

    # Запускаем обучение
    trainer.fit(config['training']['epochs'])

    writer.close()

if __name__ == '__main__':
    main()

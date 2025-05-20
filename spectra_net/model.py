import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAttentionNet(nn.Module):
    """
    Нейросеть с 1D сверточным слоем и многоголовым вниманием
    для классификации и регрессии спектров.
    """

    def __init__(self, num_classes=2, input_channels=2, conv_channels1=32, conv_channels2=64,
                 kernel_size1=7, kernel_size2=5, attn_heads=4, dropout=0.5):
        super().__init__()

        # Классификационный блок: 2 свёртки + BN + ReLU
        self.conv1 = nn.Conv1d(input_channels, conv_channels1, kernel_size=kernel_size1, padding=kernel_size1 // 2)
        self.bn1 = nn.BatchNorm1d(conv_channels1)
        self.conv2 = nn.Conv1d(conv_channels1, conv_channels2, kernel_size=kernel_size2, padding=kernel_size2 // 2)
        self.bn2 = nn.BatchNorm1d(conv_channels2)

        # MultiheadAttention после свёрток
        self.attn = nn.MultiheadAttention(embed_dim=conv_channels2, num_heads=attn_heads, batch_first=True)

        # Линейный слой для классификации (после attention)
        self.fc_shared = nn.Linear(conv_channels2, conv_channels2)
        self.dropout = nn.Dropout(dropout)
        self.fc_class = nn.Linear(conv_channels2, num_classes)

        # Регрессия: отдельный путь с свёртками по необработанному спектру (x_raw)
        self.regressor = nn.Sequential(
            nn.Conv1d(input_channels, conv_channels1, kernel_size=kernel_size1, padding=kernel_size1 // 2),
            nn.BatchNorm1d(conv_channels1),
            nn.ReLU(),
            nn.Conv1d(conv_channels1, conv_channels2, kernel_size=kernel_size2, padding=kernel_size2 // 2),
            nn.BatchNorm1d(conv_channels2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(conv_channels1),  # фиксируем длину выхода для линейных слоёв
            nn.Flatten(),
            nn.Linear(conv_channels2 * conv_channels1, conv_channels2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels2, 1)
        )


    def forward(self, x_masked, x_raw=None):
        """
        Args:
            x_masked: тензор [B, 2, L] — спектр + маска
            x_raw: тензор [B, 2, L] — необработанный спектр (для регрессии)

        Returns:
            output_cls: [B, num_classes] — классификация
            output_reg: [B, 1] — регрессия
            attention_map: [B, attn_heads, L] — карта внимания
        """
        B, C, L = x_masked.shape

        # Классификационный путь: две свёртки + BN + ReLU
        x = self.conv1(x_masked)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)  # [B, conv_channels2, L]

        # Перемещаем для attention: [B, L, conv_channels2]
        x = x.permute(0, 2, 1)

        # MultiheadAttention (query=key=value)
        attn_output, attn_weights = self.attn(x, x, x)
        # attn_weights: [B, attn_heads, L, L]

        # Усредняем по ключам, получаем [B, attn_heads, L]
        attention_map = attn_weights.mean(dim=2)

        # Классификация: берем выход attention для каждого элемента последовательности,
        # проецируем и усредняем по длине
        x_cls = self.fc_shared(attn_output)  # [B, L, conv_channels2]
        x_cls = torch.relu(x_cls)
        x_cls = self.dropout(x_cls)

        # Усредняем по длине последовательности для классификации
        x_cls = x_cls.mean(dim=1)  # [B, conv_channels2]

        output_cls = self.fc_class(x_cls)  # [B, num_classes]

        # Регрессия: отдельный путь по необработанному спектру
        if x_raw is not None:
            output_reg = self.regressor(x_raw).squeeze(1)  # [B]
        else:
            # Если необработанный спектр не подан, регрессия возвращает None
            output_reg = None

        return output_cls, output_reg, attention_map

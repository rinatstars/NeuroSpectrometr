import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """
    Комбинированная функция потерь:
    CrossEntropy для классификации + MSE для регрессии
    с настраиваемыми весами.
    """

    def __init__(self, class_weight=1.0, reg_weight=1.0):
        super().__init__()
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, output_cls, output_reg, target_cls, target_reg):
        cls_loss = self.ce(output_cls, target_cls)
        reg_loss = self.mse(output_reg, target_reg)
        loss = self.class_weight * cls_loss + self.reg_weight * reg_loss
        return loss, cls_loss, reg_loss

import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import io

class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device,
                 save_dir='./checkpoints', writer=None, class_weight=1.0, reg_weight=1.0):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.writer = writer
        self.class_weight = class_weight
        self.reg_weight = reg_weight


        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        losses = 0
        correct = 0
        total = 0

        loop = tqdm(self.dataloaders['train'], desc=f"Train Epoch {epoch}")

        for batch_idx, batch in enumerate(loop):
            x_masked = batch['masked_spectrum'].to(self.device)
            x_raw = batch['raw_spectrum'].to(self.device)
            y_cls = batch['label'].to(self.device)
            y_reg = batch['intensity'].to(self.device)

            self.optimizer.zero_grad()
            out_cls, out_reg, attn_map = self.model(x_masked, x_raw)

            loss_out, cls_loss, reg_loss = self.criterion(out_cls, out_reg, y_cls, y_reg)

            loss_reg = (torch.abs((y_reg - out_reg) / (y_reg + 1e-8))).mean()
            loss = loss_out + self.reg_weight * loss_reg

            loss.backward()
            self.optimizer.step()

            losses += loss_out.item()
            preds = out_cls.argmax(dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

            loop.set_postfix(loss=loss.item(), accuracy=correct/total)

        avg_loss = losses / len(self.dataloaders['train'])
        accuracy = correct / total
        return avg_loss, accuracy, attn_map

    def validate_epoch(self, epoch):
        self.model.eval()
        losses = 0
        correct = 0
        total = 0

        with torch.no_grad():
            loop = tqdm(self.dataloaders['val'], desc=f"Validate Epoch {epoch}")
            for batch_idx, batch in enumerate(loop):
                x_masked = batch['masked_spectrum'].to(self.device)
                x_raw = batch['raw_spectrum'].to(self.device)
                y_cls = batch['label'].to(self.device)
                y_reg = batch['intensity'].to(self.device)

                out_cls, out_reg, attn_map = self.model(x_masked, x_raw)

                loss_out, cls_loss, reg_loss = self.criterion(out_cls, out_reg, y_cls, y_reg)

                loss_reg = (torch.abs((y_reg - out_reg) / (y_reg + 1e-8))).mean()
                loss = loss_out + self.reg_weight * loss_reg

                losses += loss_out.item()
                preds = out_cls.argmax(dim=1)
                correct += (preds == y_cls).sum().item()
                total += y_cls.size(0)

                loop.set_postfix(loss=loss.item(), accuracy=correct/total)

        avg_loss = losses / len(self.dataloaders['val'])
        accuracy = correct / total
        return avg_loss, accuracy, attn_map

    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved at {path}")

    def fit(self, epochs):
        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_attention_map = self.train_epoch(epoch)
            val_loss, val_acc, val_attention_map = self.validate_epoch(epoch)

            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # лог в SummaryWriter
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch)

def plot_spectrum_with_mask(spectrum, mask, epoch):
    plt.figure(figsize=(8, 3))
    plt.plot(spectrum, label="Spectrum")
    plt.plot(mask, label="Mask")
    plt.legend()
    plt.title(f"Spectrum and Mask at Epoch {epoch}")

    # Сохраняем фигуру в байтовый поток для TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

"""
阶段4：模型训练与可视化
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from dataset import get_data_loaders
from model import ImageCaptioningModel
from vocab import Vocabulary


# ==================== 配置 ====================
CONFIG = {
    # 数据
    'data_dir': 'data/flickr8k',
    'token_file': 'data/flickr8k/Flickr8k.token.txt',
    'train_file': 'data/flickr8k/Flickr_8k.trainImages.txt',
    'val_file': 'data/flickr8k/Flickr_8k.devImages.txt',
    'max_caption_len': 35,

    # 模型
    'embed_size': 256,
    'attention_dim': 256,
    'decoder_dim': 512,
    'dropout': 0.5,
    'fine_tune_encoder': False,

    # 训练
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 3e-4,
    'num_workers': 2,
    'grad_clip': 5.0,

    # 保存
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
}

os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
os.makedirs(CONFIG['log_dir'], exist_ok=True)
os.makedirs('figures', exist_ok=True)
writer = SummaryWriter(CONFIG['log_dir'])


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f'Epoch {epoch+1} [Train]', leave=False)

    for images, captions, lengths in loop:
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        predictions, _, decode_lengths, _ = model(images, captions, lengths)

        # 逐样本计算损失
        loss = 0
        for i in range(predictions.size(0)):
            dlen = decode_lengths[i]
            if dlen > 0:
                loss += criterion(
                    predictions[i, :dlen, :],
                    captions[i, 1:dlen+1]
                )
        loss /= predictions.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=f'{loss.item():.3f}')

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0

    for images, captions, lengths in tqdm(loader, desc='[Val]', leave=False):
        images = images.to(device)
        captions = captions.to(device)

        predictions, _, decode_lengths, _ = model(images, captions, lengths)

        loss = 0
        for i in range(predictions.size(0)):
            dlen = decode_lengths[i]
            if dlen > 0:
                loss += criterion(
                    predictions[i, :dlen, :],
                    captions[i, 1:dlen+1]
                )
        loss /= predictions.size(0)
        total_loss += loss.item()

    return total_loss / len(loader)


def plot_loss_curves(train_losses, val_losses, save_path='figures/loss_curve.png'):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.title('Training and Validation Loss Curves', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  => 损失曲线已保存: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print("=" * 60)

    # 加载词汇表
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    VOCAB_SIZE = len(vocab)
    print(f"词汇表大小: {VOCAB_SIZE}")

    # 数据加载器
    print("\n加载数据...")
    train_loader, val_loader = get_data_loaders(
        data_dir=CONFIG['data_dir'],
        token_file=CONFIG['token_file'],
        train_file=CONFIG['train_file'],
        val_file=CONFIG['val_file'],
        vocab=vocab,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        max_len=CONFIG['max_caption_len']
    )

    # 模型
    print("\n初始化模型...")
    model = ImageCaptioningModel(
        embed_size=CONFIG['embed_size'],
        attention_dim=CONFIG['attention_dim'],
        decoder_dim=CONFIG['decoder_dim'],
        vocab_size=VOCAB_SIZE,
        dropout=CONFIG['dropout'],
        fine_tune=CONFIG['fine_tune_encoder']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}")

    # 损失与优化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(CONFIG['num_epochs']):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 日志
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{CONFIG["checkpoint_dir"]}/best_model.pth')
            print(f"  ✓ 最佳模型已保存 (Val Loss: {best_val_loss:.4f})")

        # 每10轮保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'{CONFIG["checkpoint_dir"]}/checkpoint_epoch{epoch+1}.pth')

    total_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {total_time/60:.1f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.4f}")

    # 绘制损失曲线
    plot_loss_curves(train_losses, val_losses)

    writer.close()
    print("TensorBoard日志已保存到: logs/")


if __name__ == '__main__':
    main()

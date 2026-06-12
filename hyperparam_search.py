"""
阶段5：超参数调优 — 多组参数对比
"""
import os
import torch
import pickle
import itertools
import json
from train import train_one_epoch, validate
from dataset import get_data_loaders
from model import ImageCaptioningModel
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 搜索空间
SEARCH_SPACE = [
    {'lr': 3e-4, 'batch_size': 32, 'dropout': 0.5, 'decoder_dim': 512, 'name': 'Baseline'},
    {'lr': 1e-3, 'batch_size': 32, 'dropout': 0.5, 'decoder_dim': 512, 'name': 'Higher LR'},
    {'lr': 3e-4, 'batch_size': 64, 'dropout': 0.5, 'decoder_dim': 512, 'name': 'Larger Batch'},
    {'lr': 3e-4, 'batch_size': 32, 'dropout': 0.3, 'decoder_dim': 512, 'name': 'Lower Dropout'},
    {'lr': 3e-4, 'batch_size': 32, 'dropout': 0.5, 'decoder_dim': 256, 'name': 'Smaller Decoder'},
]


def run_experiment(config, vocab, base_config):
    """运行一组实验"""
    print(f"\n{'='*60}")
    print(f"实验: {config['name']}")
    print(f"参数: lr={config['lr']}, bs={config['batch_size']}, "
          f"dropout={config['dropout']}, dec_dim={config['decoder_dim']}")
    print(f"{'='*60}")

    train_loader, val_loader = get_data_loaders(
        data_dir=base_config['data_dir'],
        token_file=base_config['token_file'],
        train_file=base_config['train_file'],
        val_file=base_config['val_file'],
        vocab=vocab,
        batch_size=config['batch_size'],
        max_len=base_config['max_caption_len']
    )

    model = ImageCaptioningModel(
        embed_size=base_config['embed_size'],
        attention_dim=base_config['attention_dim'],
        decoder_dim=config['decoder_dim'],
        vocab_size=len(vocab),
        dropout=config['dropout'],
        fine_tune=False
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # 训练10轮作为快速评估
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return {
        'name': config['name'],
        'config': config,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


def main():
    print("=" * 60)
    print("阶段5：超参数调优")
    print("=" * 60)

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    base_config = {
        'data_dir': 'data/flickr8k',
        'token_file': 'data/flickr8k/Flickr8k.token.txt',
        'train_file': 'data/flickr8k/Flickr_8k.trainImages.txt',
        'val_file': 'data/flickr8k/Flickr_8k.devImages.txt',
        'max_caption_len': 35,
        'embed_size': 256,
        'attention_dim': 256,
    }

    results = []
    for config in SEARCH_SPACE:
        result = run_experiment(config, vocab, base_config)
        results.append(result)

    # 汇总结果
    print("\n" + "=" * 80)
    print("超参数搜索结果汇总")
    print("=" * 80)
    print(f"{'实验':<20} {'LR':<10} {'BS':<6} {'Dropout':<8} {'DecDim':<8} {'Best Val Loss':<14} {'Final Val Loss'}")
    print("-" * 80)
    for r in results:
        c = r['config']
        print(f"{r['name']:<20} {c['lr']:<10.0e} {c['batch_size']:<6} "
              f"{c['dropout']:<8} {c['decoder_dim']:<8} "
              f"{r['best_val_loss']:<14.4f} {r['final_val_loss']:.4f}")

    # 保存结果
    with open('hyperparam_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n结果已保存至: hyperparam_results.json")

    # 找出最佳
    best = min(results, key=lambda x: x['best_val_loss'])
    print(f"\n🏆 最佳配置: {best['name']} (Val Loss: {best['best_val_loss']:.4f})")


if __name__ == '__main__':
    main()

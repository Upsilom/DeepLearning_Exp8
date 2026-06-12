"""
阶段2-2：PyTorch Dataset 数据加载器
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pickle
from vocab import Vocabulary


class Flickr8kDataset(Dataset):
    """Flickr8k 图像描述数据集"""
    def __init__(self, data_dir, token_file, split_file, vocab, 
                 transform=None, max_len=35, phase='train'):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'Flicker8k_Dataset')
        self.vocab = vocab
        self.max_len = max_len
        self.phase = phase

        # 加载该划分的图片名
        with open(split_file, 'r') as f:
            self.split_imgs = set(line.strip() for line in f)

        # 加载所有描述，只保留该划分的
        self.samples = []
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_caption_id, caption = parts
                    img_name = img_caption_id.split('#')[0]
                    if img_name in self.split_imgs:
                        self.samples.append((img_name, caption))

        # 数据增强（训练集） vs 基础变换（验证/测试）
        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                       saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]

        # 加载并变换图像
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"  警告: 无法加载 {img_path}: {e}")
            # 返回一张黑图作为fallback
            image = torch.zeros(3, 224, 224)

        # 描述 -> token IDs
        token_ids = [self.vocab.stoi["<SOS>"]] + \
                    self.vocab.numericalize(caption) + \
                    [self.vocab.stoi["<EOS>"]]

        # 截断
        token_ids = token_ids[:self.max_len]

        # 计算实际长度（不含填充）
        actual_len = len(token_ids)

        # 填充到 max_len
        padded = token_ids + [self.vocab.stoi["<PAD>"]] * (self.max_len - len(token_ids))

        return image, torch.tensor(padded, dtype=torch.long), torch.tensor(actual_len, dtype=torch.long)


def get_data_loaders(data_dir, token_file, train_file, val_file, vocab,
                     batch_size=32, num_workers=2, max_len=35):
    """创建训练和验证数据加载器"""
    train_dataset = Flickr8kDataset(
        data_dir, token_file, train_file, vocab,
        max_len=max_len, phase='train'
    )
    val_dataset = Flickr8kDataset(
        data_dir, token_file, val_file, vocab,
        max_len=max_len, phase='val'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  训练集: {len(train_dataset)} 条 (批数={len(train_loader)})")
    print(f"  验证集: {len(val_dataset)} 条 (批数={len(val_loader)})")

    return train_loader, val_loader


if __name__ == '__main__':
    # 快速测试
    print("测试数据加载器...")
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    train_loader, val_loader = get_data_loaders(
        data_dir='data/flickr8k',
        token_file='data/flickr8k/Flickr8k.token.txt',
        train_file='data/flickr8k/Flickr_8k.trainImages.txt',
        val_file='data/flickr8k/Flickr_8k.devImages.txt',
        vocab=vocab,
        batch_size=4
    )

    images, captions, lengths = next(iter(train_loader))
    print(f"  图像shape: {images.shape}")        # (4, 3, 224, 224)
    print(f"  描述shape: {captions.shape}")      # (4, 35)
    print(f"  长度: {lengths.tolist()}")

    # 解码一个样例
    cap_ids = captions[0][1:lengths[0]-1].tolist()  # 去掉<SOS>和<EOS>
    decoded = vocab.decode(cap_ids)
    print(f"  样例解码: '{decoded}'")

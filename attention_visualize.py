"""
阶段6：注意力可解释性可视化
展示模型在生成每个词时关注图像的哪些区域
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os

from model import ImageCaptioningModel
from vocab import Vocabulary

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def generate_with_attention(model, vocab, image_tensor, max_len=35):
    """生成描述并返回每步的注意力权重"""
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor.unsqueeze(0).to(DEVICE))  # (1, 196, 256)
        h, c = model.decoder.init_hidden_state(features)

        word = torch.tensor([vocab.stoi["<SOS>"]]).to(DEVICE)
        words = ['<SOS>']
        all_alphas = []

        for _ in range(max_len):
            embeddings = model.decoder.embedding(word)
            context, alpha = model.decoder.attention(features, h)

            lstm_input = torch.cat([embeddings.squeeze(1), context], dim=1)
            h, c = model.decoder.lstm_cell(lstm_input, (h, c))

            preds = model.decoder.fc(model.decoder.dropout(h))
            word = preds.argmax(dim=1)
            word_idx = word.item()

            if word_idx == vocab.stoi["<EOS>"]:
                words.append('<EOS>')
                all_alphas.append(alpha.squeeze(0).cpu().numpy())
                break

            words.append(vocab.itos.get(word_idx, '<UNK>'))
            all_alphas.append(alpha.squeeze(0).cpu().numpy())

        return words, all_alphas


def visualize_attention(image_path, model, vocab):
    """可视化注意力热力图"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)

    words, alphas = generate_with_attention(model, vocab, img_tensor)

    # 只展示有意义的词（跳过<SOS>和<EOS>）
    display_words = words[1:-1] if words[-1] == '<EOS>' else words[1:]
    display_alphas = alphas[:len(display_words)]

    if len(display_words) == 0:
        print("  未生成任何词!")
        return

    # 原始图片用于显示
    orig_img = Image.open(image_path).resize((224, 224))

    # 构建子图网格
    n_words = len(display_words)
    cols = min(5, n_words)
    rows = (n_words + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (word, alpha) in enumerate(zip(display_words, display_alphas)):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        # 重塑注意力到14×14并上采样到224×224
        alpha_2d = alpha.reshape(14, 14)
        from torch.nn.functional import interpolate
        alpha_tensor = torch.tensor(alpha_2d).unsqueeze(0).unsqueeze(0)
        alpha_upsampled = interpolate(alpha_tensor, size=(224, 224), mode='bilinear').squeeze().numpy()

        ax.imshow(orig_img)
        ax.imshow(alpha_upsampled, cmap='jet', alpha=0.5)
        ax.set_title(f'"{word}"', fontsize=11, fontweight='bold')
        ax.axis('off')

    # 隐藏多余子图
    for idx in range(n_words, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')

    plt.suptitle('Attention Visualization — Model Focus During Caption Generation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/attention_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  => 已保存: figures/attention_visualization.png")


def main():
    print("=" * 60)
    print("注意力可解释性可视化")
    print("=" * 60)

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    model = ImageCaptioningModel(
        embed_size=256, attention_dim=256, decoder_dim=512,
        vocab_size=len(vocab), dropout=0.5
    ).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=DEVICE))

    # 随机选验证集图片
    with open('data/flickr8k/Flickr_8k.devImages.txt', 'r') as f:
        val_imgs = [line.strip() for line in f.readlines()]

    sample_imgs = np.random.choice(val_imgs, 3, replace=False)

    for img_name in sample_imgs:
        img_path = os.path.join('data/flickr8k/Flicker8k_Dataset', img_name)
        print(f"\n图片: {img_name}")
        visualize_attention(img_path, model, vocab)


if __name__ == '__main__':
    main()

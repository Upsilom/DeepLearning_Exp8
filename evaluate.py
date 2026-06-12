"""
阶段6：模型评估 — 无NLTK依赖版（自带BLEU计算）
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from collections import Counter

from model import ImageCaptioningModel
from dataset import Flickr8kDataset
from vocab import Vocabulary


# ==================== 简单分词 ====================
def tokenize(text):
    for punct in ['.', ',', '?', '!', ':', ';', '-', '(', ')', '[', ']', '"', "'"]:
        text = text.replace(punct, ' ' + punct + ' ')
    return text.lower().split()


# ==================== 自实现 BLEU 计算 ====================
def ngrams(tokens, n):
    """生成n-gram"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def modified_precision(reference_tokens, candidate_tokens, n):
    """修正n-gram精度"""
    ref_ngrams = Counter(ngrams(reference_tokens, n))
    cand_ngrams = Counter(ngrams(candidate_tokens, n))

    total = sum(cand_ngrams.values())
    if total == 0:
        return 0.0

    clipped = 0
    for ngram, count in cand_ngrams.items():
        clipped += min(count, ref_ngrams.get(ngram, 0))

    return clipped / total if total > 0 else 0.0


def closest_ref_length(candidate_len, ref_lengths):
    """找最接近的参考长度"""
    return min(ref_lengths, key=lambda x: abs(x - candidate_len))


def brevity_penalty(candidate_len, ref_lengths):
    """长度惩罚"""
    closest = closest_ref_length(candidate_len, ref_lengths)
    if candidate_len > closest:
        return 1.0
    elif candidate_len == 0:
        return 0.0
    else:
        return math.exp(1 - closest / candidate_len)


def sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25)):
    """单句BLEU"""
    candidate_tokens = tokenize(candidate)
    candidate_len = len(candidate_tokens)

    ref_lengths = [len(tokenize(ref)) for ref in references]
    bp = brevity_penalty(candidate_len, ref_lengths)

    max_n = len(weights)
    precisions = []
    for n in range(1, max_n + 1):
        # 对所有reference取最大精度
        max_prec = 0.0
        all_ref_tokens = [tokenize(ref) for ref in references]
        for ref_tokens in all_ref_tokens:
            prec = modified_precision(ref_tokens, candidate_tokens, n)
            max_prec = max(max_prec, prec)
        precisions.append(max_prec)

    # 几何平均
    if any(p == 0 for p in precisions):
        return 0.0

    return bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))


def corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25)):
    """语料级BLEU"""
    scores = []
    for refs, cand in zip(references, candidates):
        score = sentence_bleu(refs, cand, weights)
        scores.append(score)
    return np.mean(scores) if scores else 0.0


# ==================== 配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
VOCAB_PATH = 'vocab.pkl'
MAX_LEN = 35

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def generate_caption(model, vocab, image_tensor, max_len=MAX_LEN):
    """生成图像描述（贪心解码）"""
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor.unsqueeze(0).to(DEVICE))
        h, c = model.decoder.init_hidden_state(features)
        word = torch.tensor([vocab.stoi["<SOS>"]]).to(DEVICE)
        caption_ids = []
        alphas = []

        for _ in range(max_len):
            embeddings = model.decoder.embedding(word)
            context, alpha = model.decoder.attention(features, h)
            lstm_input = torch.cat([embeddings.squeeze(1), context], dim=1)
            h, c = model.decoder.lstm_cell(lstm_input, (h, c))
            preds = model.decoder.fc(model.decoder.dropout(h))
            word = preds.argmax(dim=1)
            word_idx = word.item()

            if word_idx == vocab.stoi["<EOS>"]:
                break

            caption_ids.append(word_idx)
            alphas.append(alpha.squeeze(0).cpu().numpy())

        return vocab.decode(caption_ids), alphas


def compute_bleu_scores(model, vocab, num_samples=200):
    """计算验证集上的 BLEU 分数"""
    print("\n计算 BLEU 分数...")

    # 加载所有ground truth
    gt_dict = {}
    with open('data/flickr8k/Flickr8k.token.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_id, cap = parts
                img_name = img_id.split('#')[0]
                if img_name not in gt_dict:
                    gt_dict[img_name] = []
                gt_dict[img_name].append(cap)

    # 加载验证集图片名
    with open('data/flickr8k/Flickr_8k.devImages.txt', 'r') as f:
        val_imgs = [line.strip() for line in f.readlines()[:num_samples]]

    references_all = []
    hypotheses_all = []

    for img_name in tqdm(val_imgs, desc='BLEU计算'):
        img_path = os.path.join('data/flickr8k/Flicker8k_Dataset', img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            caption, _ = generate_caption(model, vocab, img_tensor)

            refs = gt_dict.get(img_name, [''])
            hypotheses_all.append(caption)
            references_all.append(refs)
        except Exception as e:
            continue

    # 计算 BLEU
    bleu1 = corpus_bleu(references_all, hypotheses_all, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references_all, hypotheses_all, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references_all, hypotheses_all, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references_all, hypotheses_all, weights=(0.25, 0.25, 0.25, 0.25))

    return {'BLEU-1': bleu1, 'BLEU-2': bleu2, 'BLEU-3': bleu3, 'BLEU-4': bleu4}


def visualize_results(model, vocab, num_samples=6):
    """可视化推理结果"""
    with open('data/flickr8k/Flickr_8k.devImages.txt', 'r') as f:
        val_imgs = [line.strip() for line in f.readlines()]

    sample_imgs = np.random.choice(val_imgs, num_samples, replace=False)

    rows = (num_samples + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 5 * rows))
    axes = axes.flatten() if num_samples > 2 else [axes]

    for ax, img_name in zip(axes, sample_imgs):
        img_path = os.path.join('data/flickr8k/Flicker8k_Dataset', img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)

        caption, _ = generate_caption(model, vocab, img_tensor)

        ax.imshow(Image.open(img_path))
        ax.set_title(f'Generated: {caption}', fontsize=10, color='blue',
                     pad=10, wrap=True)
        ax.axis('off')

    for ax in axes[num_samples:]:
        ax.axis('off')

    plt.suptitle('Model Inference Results on Validation Set', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/inference_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  => 已保存: figures/inference_results.png")


def main():
    print("=" * 60)
    print("阶段6：模型评估")
    print("=" * 60)

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    model = ImageCaptioningModel(
        embed_size=256, attention_dim=256, decoder_dim=512,
        vocab_size=len(vocab), dropout=0.5
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print(f"模型已加载自: {CHECKPOINT_PATH}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {params:.1f}M")

    # 推理速度
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    import time
    for _ in range(10):
        _ = generate_caption(model, vocab, dummy.squeeze(0))
    times = []
    for _ in range(30):
        start = time.time()
        _ = generate_caption(model, vocab, dummy.squeeze(0))
        times.append(time.time() - start)
    print(f"平均推理时间: {np.mean(times)*1000:.0f}ms | FPS: {1/np.mean(times):.1f}")

    # BLEU
    bleu_scores = compute_bleu_scores(model, vocab)
    print("\n【BLEU 评估结果】")
    for k, v in bleu_scores.items():
        print(f"  {k}: {v:.4f}")

    # 可视化
    print("\n生成推理可视化...")
    visualize_results(model, vocab)

    print("\n评估完成!")


if __name__ == '__main__':
    main()

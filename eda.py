"""
阶段1：探索性数据分析 (EDA)
Flickr8k 数据集分析 — 无NLTK依赖版
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

# ==================== 简单分词函数（替代NLTK） ====================
def tokenize(text):
    """简单按空格分词，处理基本标点"""
    # 将常见标点前后加空格再split
    for punct in ['.', ',', '?', '!', ':', ';', '-', '(', ')', '[', ']', '"', "'"]:
        text = text.replace(punct, ' ' + punct + ' ')
    return text.lower().split()


# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'

DATA_DIR = 'data/flickr8k'
IMAGE_DIR = os.path.join(DATA_DIR, 'Flicker8k_Dataset')
TOKEN_FILE = os.path.join(DATA_DIR, 'Flickr8k.token.txt')
TRAIN_FILE = os.path.join(DATA_DIR, 'Flickr_8k.trainImages.txt')
VAL_FILE = os.path.join(DATA_DIR, 'Flickr_8k.devImages.txt')
TEST_FILE = os.path.join(DATA_DIR, 'Flickr_8k.testImages.txt')

os.makedirs('figures', exist_ok=True)

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("阶段1：Flickr8k 数据集 EDA 分析")
print("=" * 60)

# 加载描述文件
captions_dict = {}
with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            img_caption_id, caption = parts
            img_name = img_caption_id.split('#')[0]
            if img_name not in captions_dict:
                captions_dict[img_name] = []
            captions_dict[img_name].append(caption)

# 加载划分
def load_split(filename):
    with open(filename, 'r') as f:
        return set(line.strip() for line in f)

train_imgs = load_split(TRAIN_FILE)
val_imgs = load_split(VAL_FILE)
test_imgs = load_split(TEST_FILE)

all_img_names = list(captions_dict.keys())

# ==================== 2. 基本统计 ====================
print("\n【基本统计】")
print(f"  总图片数: {len(all_img_names)}")
print(f"  总描述数: {sum(len(v) for v in captions_dict.values())}")
print(f"  每图描述数: {len(list(captions_dict.values())[0])}")
print(f"  训练集图片: {len(train_imgs)} ({len(train_imgs)/len(all_img_names)*100:.1f}%)")
print(f"  验证集图片: {len(val_imgs)} ({len(val_imgs)/len(all_img_names)*100:.1f}%)")
print(f"  测试集图片: {len(test_imgs)} ({len(test_imgs)/len(all_img_names)*100:.1f}%)")

# ==================== 3. 描述长度分析 ====================
all_lengths = []
for captions in captions_dict.values():
    for cap in captions:
        tokens = tokenize(cap)
        all_lengths.append(len(tokens))

print(f"\n【描述长度统计】")
print(f"  均值: {np.mean(all_lengths):.1f} 词")
print(f"  中位数: {np.median(all_lengths):.1f} 词")
print(f"  标准差: {np.std(all_lengths):.2f}")
print(f"  最小: {np.min(all_lengths)}  最大: {np.max(all_lengths)}")
print(f"  95%分位数: {np.percentile(all_lengths, 95):.0f}")

# 绘制描述长度分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1: 直方图
axes[0].hist(all_lengths, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(np.mean(all_lengths), color='red', linestyle='--',
                linewidth=2, label=f'Mean = {np.mean(all_lengths):.1f}')
axes[0].axvline(np.median(all_lengths), color='orange', linestyle='--',
                linewidth=2, label=f'Median = {np.median(all_lengths):.1f}')
axes[0].set_xlabel('Caption Length (words)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Caption Lengths', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# 子图2: 累计分布
sorted_lengths = np.sort(all_lengths)
cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
axes[1].plot(sorted_lengths, cdf, 'b-', linewidth=2)
axes[1].axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95th percentile')
axes[1].axvline(np.percentile(all_lengths, 95), color='red', linestyle=':', alpha=0.7)
axes[1].set_xlabel('Caption Length (words)', fontsize=12)
axes[1].set_ylabel('Cumulative Probability', fontsize=12)
axes[1].set_title('CDF of Caption Lengths', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/caption_length_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("  => 已保存: figures/caption_length_distribution.png")

# ==================== 4. 词汇分析 ====================
all_tokens = []
for captions in captions_dict.values():
    for cap in captions:
        all_tokens.extend(tokenize(cap))

word_counter = Counter(all_tokens)
vocab_size = len(word_counter)
print(f"\n【词汇统计】")
print(f"  总词汇量: {vocab_size}")
print(f"  出现1次的词: {sum(1 for c in word_counter.values() if c == 1)} ({sum(1 for c in word_counter.values() if c == 1)/vocab_size*100:.1f}%)")
print(f"  出现≥5次的词: {sum(1 for c in word_counter.values() if c >= 5)}")

# 绘制词频Top30
top30 = word_counter.most_common(30)
words, freqs = zip(*top30)

plt.figure(figsize=(14, 6))
bars = plt.bar(words, freqs, color='steelblue', edgecolor='white')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.xlabel('Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Top 30 Most Frequent Words in Flickr8k Captions', fontsize=14)
bars[0].set_color('coral')
for bar, freq in zip(bars[:5], freqs[:5]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             str(freq), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('figures/word_frequency_top30.png', dpi=150, bbox_inches='tight')
plt.show()
print("  => 已保存: figures/word_frequency_top30.png")

# ==================== 5. 图像尺寸分析 ====================
sample_sizes = []
for img_name in list(all_img_names)[:500]:
    try:
        img_path = os.path.join(IMAGE_DIR, img_name)
        with Image.open(img_path) as img:
            sample_sizes.append(img.size)
    except:
        pass

widths, heights = zip(*sample_sizes)
print(f"\n【图像尺寸统计(采样500张)】")
print(f"  宽度: 均值={np.mean(widths):.0f}, 范围=[{np.min(widths)}, {np.max(widths)}]")
print(f"  高度: 均值={np.mean(heights):.0f}, 范围=[{np.min(heights)}, {np.max(heights)}]")

# ==================== 6. 样本可视化 ====================
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
sample_imgs = np.random.choice(all_img_names, 8, replace=False)

for ax, img_name in zip(axes.flatten(), sample_imgs):
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')
    cap = captions_dict[img_name][0][:60] + '...' if len(captions_dict[img_name][0]) > 60 else captions_dict[img_name][0]
    ax.set_title(cap, fontsize=7, wrap=True)

plt.suptitle('Flickr8k Sample Images with Captions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("  => 已保存: figures/sample_images.png")

# ==================== 7. 数据集划分验证 ====================
assert len(train_imgs & val_imgs) == 0, "训练集和验证集有重叠!"
assert len(train_imgs & test_imgs) == 0, "训练集和测试集有重叠!"
assert len(val_imgs & test_imgs) == 0, "验证集和测试集有重叠!"

all_split = len(train_imgs) + len(val_imgs) + len(test_imgs)
print(f"\n【数据集划分验证】")
print(f"  ✓ 训练/验证/测试无重叠")
print(f"  划分总计: {all_split} (应有{len(all_img_names)}张)")

print("\n" + "=" * 60)
print("EDA 分析完成！请查看 figures/ 目录下的图表。")
print("=" * 60)

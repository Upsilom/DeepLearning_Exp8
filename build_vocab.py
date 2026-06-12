"""
阶段2-1：构建词汇表并保存
"""
import pickle
import os
from vocab import Vocabulary, tokenize

DATA_DIR = 'data/flickr8k'
TOKEN_FILE = os.path.join(DATA_DIR, 'Flickr8k.token.txt')
TRAIN_FILE = os.path.join(DATA_DIR, 'Flickr_8k.trainImages.txt')


def main():
    print("=" * 60)
    print("阶段2-1：构建词汇表")
    print("=" * 60)

    with open(TRAIN_FILE, 'r') as f:
        train_img_names = set(line.strip() for line in f)

    with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    train_captions = []
    for line in all_lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            img_caption_id, caption = parts
            img_name = img_caption_id.split('#')[0]
            if img_name in train_img_names:
                train_captions.append(caption)

    print(f"  训练集描述数: {len(train_captions)}")

    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(train_captions)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print(f"  ✓ 词汇表已保存至 vocab.pkl ({len(vocab)} 个词)")

    test_cap = "a dog runs on the grass"
    ids = vocab.numericalize(test_cap)
    decoded = vocab.decode(ids)
    print(f"  测试: '{test_cap}' -> {ids} -> '{decoded}'")
    print("=" * 60)


if __name__ == '__main__':
    main()

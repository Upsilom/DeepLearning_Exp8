"""
词汇表模块 — 独立的 Vocabulary 类
"""
from collections import Counter


def tokenize(text):
    """简单按空格分词"""
    for punct in ['.', ',', '?', '!', ':', ';', '-', '(', ')', '[', ']', '"', "'"]:
        text = text.replace(punct, ' ' + punct + ' ')
    return text.lower().split()


class Vocabulary:
    """词汇表类"""
    def __init__(self, freq_threshold=5):
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, captions_list):
        word_counter = Counter()
        for cap in captions_list:
            word_counter.update(tokenize(cap))

        added = 0
        skipped = 0
        for word, count in word_counter.most_common():
            if count >= self.freq_threshold:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx
                added += 1
            else:
                skipped += 1

        print(f"  词汇表构建完成: 保留 {added} 词, 丢弃 {skipped} 低频词")
        print(f"  总词汇量(含特殊token): {len(self.itos)}")

    def numericalize(self, text):
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenize(text)
        ]

    def decode(self, ids):
        return ' '.join([
            self.itos.get(id, "<UNK>")
            for id in ids
            if id not in [self.stoi["<PAD>"], self.stoi["<SOS>"], self.stoi["<EOS>"]]
        ])

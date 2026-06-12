"""
阶段3：编码器-解码器 + 注意力机制
参考: Show, Attend and Tell (Xu et al., 2015)
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torchvision.models as models
from vocab import Vocabulary



class EncoderCNN(nn.Module):
    """ResNet-101 编码器，提取图像全局特征"""
    def __init__(self, embed_size=256, fine_tune=False):
        super().__init__()
        # 使用预训练 ResNet-101
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        # 去掉最后的池化和全连接
        modules = list(resnet.children())[:-2]  # 保留到 avgpool 之前
        self.resnet = nn.Sequential(*modules)
        # 自适应池化到固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        # 投影到 embed_size
        self.fc = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(0.3)

        # 冻结/微调控制
        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        返回: (B, 196, embed_size) — 14×14=196 个空间位置
        """
        features = self.resnet(images)          # (B, 2048, 7, 7) 或类似
        features = self.adaptive_pool(features)  # (B, 2048, 14, 14)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # (B, 14, 14, 2048)
        features = features.reshape(B, H * W, C) # (B, 196, 2048)
        features = self.fc(features)             # (B, 196, embed_size)
        features = self.dropout(features)
        return features


class Attention(nn.Module):
    """Bahdanau 加法注意力"""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.full_att = nn.Linear(attention_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out:  (B, L, encoder_dim)
        decoder_hidden: (B, decoder_dim)
        返回: context (B, encoder_dim), alpha (B, L)
        """
        att1 = self.encoder_att(encoder_out)          # (B, L, att_dim)
        att2 = self.decoder_att(decoder_hidden)        # (B, att_dim)
        att = self.full_att(torch.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (B, L)
        alpha = self.softmax(att)                      # (B, L)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha


class DecoderWithAttention(nn.Module):
    """LSTM解码器 + 注意力机制"""
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=256, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 注意力
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # LSTM
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # 输出层
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # 隐藏状态初始化
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

    def init_hidden_state(self, encoder_out):
        """用编码器特征的均值初始化LSTM隐藏状态"""
        mean_encoder = encoder_out.mean(dim=1)   # (B, encoder_dim)
        h = self.init_h(mean_encoder)            # (B, decoder_dim)
        c = self.init_c(mean_encoder)            # (B, decoder_dim)
        return h, c

    def forward(self, encoder_out, captions, caption_lengths):
        """
        encoder_out: (B, L, encoder_dim)
        captions: (B, max_len)
        caption_lengths: (B,) — 实际长度
        """
        device = encoder_out.device
        batch_size = encoder_out.size(0)

        # 嵌入所有词 (用于 teacher forcing)
        embeddings = self.embedding(captions)  # (B, max_len, embed_dim)

        # 初始化 LSTM 状态
        h, c = self.init_hidden_state(encoder_out)

        # 预测长度 = 实际长度 - 1（不需要预测 <SOS>）
        decode_lengths = [int(l) - 1 for l in caption_lengths]
        max_decode_len = max(decode_lengths)

        # 存储预测和注意力权重
        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_decode_len, encoder_out.size(1)).to(device)

        # 逐时间步解码
        for t in range(max_decode_len):
            # 哪些样本还需要继续
            mask = torch.tensor([l > t for l in decode_lengths], device=device)
            batch_size_t = mask.sum().item()
            if batch_size_t == 0:
                break

            # 获取当前有效的样本
            idx = mask.nonzero(as_tuple=True)[0]
            enc_t = encoder_out[idx]     # (B_t, L, enc_dim)
            h_t = h[idx]                 # (B_t, dec_dim)
            c_t = c[idx]                 # (B_t, dec_dim)
            emb_t = embeddings[idx, t]   # (B_t, embed_dim)

            # 计算注意力
            context, alpha = self.attention(enc_t, h_t)
            alphas[idx, t, :] = alpha

            # LSTM 一步
            lstm_input = torch.cat([emb_t, context], dim=1)
            h_next, c_next = self.lstm_cell(lstm_input, (h_t, c_t))

            # 更新状态
            h[idx] = h_next
            c[idx] = c_next

            # 预测
            preds = self.fc(self.dropout(h_next))
            predictions[idx, t, :] = preds

        return predictions, captions, decode_lengths, alphas


class ImageCaptioningModel(nn.Module):
    """完整模型: CNN编码器 + LSTM解码器 + 注意力"""
    def __init__(self, embed_size=256, attention_dim=256, decoder_dim=512,
                 vocab_size=10000, encoder_dim=256, dropout=0.5, fine_tune=False):
        super().__init__()
        self.encoder = EncoderCNN(embed_size=embed_size, fine_tune=fine_tune)
        self.decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=embed_size,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            dropout=dropout
        )

    def forward(self, images, captions, caption_lengths):
        features = self.encoder(images)   # (B, 196, embed_size)
        return self.decoder(features, captions, caption_lengths)


# ==================== 测试 ====================
if __name__ == '__main__':
    import pickle
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    model = ImageCaptioningModel(
        embed_size=256, attention_dim=256, decoder_dim=512,
        vocab_size=len(vocab), fine_tune=False
    )

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")

    # 前向传播测试
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_captions = torch.randint(0, len(vocab), (2, 35))
    dummy_lengths = torch.tensor([15, 20])

    predictions, _, decode_lengths, alphas = model(dummy_images, dummy_captions, dummy_lengths)
    print(f"  输入: 图像={dummy_images.shape}, 描述={dummy_captions.shape}")
    print(f"  输出: 预测={predictions.shape}, 注意力={alphas.shape}")
    print(f"  ✓ 模型测试通过!")

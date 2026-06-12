# 深度学习实验8：图像描述 (Image Captioning)

基于 Encoder-Decoder + Attention 机制的图像描述模型，参考 Show, Attend and Tell 论文。

## 环境配置

```bash
conda create -n image_caption python=3.9 -y
conda activate image_caption
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow nltk tensorboard

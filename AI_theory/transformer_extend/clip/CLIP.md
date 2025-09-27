# CLIP
稀土掘金：[CV大模型系列之：多模态经典之作CLIP，探索图文结合的奥秘](https://juejin.cn/post/7264503343996747830?searchId=20250926120931205C9E75390C39424212)<br>
codes: [CLIP](https://github.com/openai/CLIP)

# Q: Vit & Clip
## Q1: Vit
问题 1：我们正在使用一个 ViT（Vision Transformer）模型来分析一组图片。给定以下参数：
-  图片大小：256x256x3 （宽 x 高 x 颜色通道）
-  Patch 大小：16x16
-  Transformer 的 encoder layers：12
-  Embedding 维度：768
-  MLP（多层感知器）的隐藏层尺寸：3072
-    头的数量（number of heads）在 multi-head self-attention 机制中：12

请计算以下几个问题：
1. 计算每张图片被分割成多少个 patches，并求出每个 patch 的维度。
2. 在 multi-head self-attention 机制中，每个“头”处理的维度是多少？
3. 如果我们想要计算一次前向传播，至少需要多少个权重参数？（假设 bias 和层归一化
（layer normalization）参数不计算在内，而且只计算一个 encoder layer 里的参数）

参考答案：<br>
<font color=red>1.计算每张图片被分割成多少个 patches，并求出每个 patch 的维度。</font><br>
答案：256个patches、每个patch为768维
分析过程：依据图片大小信息256x256x3与Patch 大小16x16可以得出，图片在长、宽方向各自被分成16份，共会形成(256/16) x (256/16) = 16 x 16 = 256个patches。

<font color=red>2.在 multi-head self-attention 机制中，每个“头”处理的维度是多少？</font><br>
答案：64
分析过程：每个“头”处理的维度是 数量/头的数量=768/12=64

<font color=red>3.如果我们想要计算一次前向传播，至少需要多少个权重参数？（假设 bias 和层归一化（layer normalization）参数不计算在内，而且只计算一个 encoder layer 里的参数</font><br>

答案：该设定下计算参数总量Self-Attention + MLP = 2359296 + 4718592 = 7077888，如果强调embedding嵌入层也需要计算的话=257*768+4*768*768+2*768*3072=7275264

分析过程 共有3步：<br>
按照题目条件，不考虑bias和norm的参数，一个encoder需要计算self-attention及MLP部分的参数，分别计算并加和处理：
（1）如果要把patchembedding计入的话，输入embedding（256，768）+[cls]( 1, 768 ) = ( 257 , 768 )  共197376参数。
（2）self-attention的模块参数有Q、K、V的权重矩阵Wq、Wk、Wv以及输出权重矩阵Wo，这4个矩阵形状相同，都为[768,768]，所以self-attention的计算参数是 4*768*768=2359296
（3）MLP模块按最原始的transformer架构来看，需要计算两个线性层的参数，一个线性层负责将维度从768映射到3072，权重矩阵参数是[768，3072]。一个线性层将维度从3072映射到768, 权重矩阵参数是[3072，768]。参数计算量总和为2*768*3072=4718592


## 问题2：训练一个简化版的 CLIP 
模型背景:<br>
CLIP (Contrastive Language-Image Pre-training) 是一个多模态视觉和语言模型。它通过联合学习图像和文本的表示，可以在各种视觉和语言任务上进行零样本学习。请你使用PyTorch 实现一个简化版本的 CLIP 模型。

任务:

1. 数据预处理： 创建一个数据预处理流程，可以从数据集中读取图像和对应的文本描述， 并将它们转换成适合模型训练的格式。

2. 模型构建：构建一个简化的 CLIP 模型，该模型应包括两个主要组件：一个用于图像特征提取的 CNN（可以使用预训练模型）和一个用于处理文本输入的 Transformer  编码器。 

3. 对比损失函数：实现 CLIP 的对比损失函数。 

4. 训练循环：使用train_loop 函数，在主函数初始化模型、优化器后，实现训练循环。

参考答案（deepseek写的）
```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
from PIL import Image
import os

# ----------------------
# 1. 数据预处理
# ----------------------
class CLIPDataset(Dataset):
    def __init__(self, image_dir, text_descriptions, image_size=224):
        """
        image_dir: 图像文件夹路径
        text_descriptions: 字典 {image_filename: "text description"}
        """
        self.image_paths = [os.path.join(image_dir, fname) for fname in text_descriptions.keys()]
        self.texts = list(text_descriptions.values())
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 文本tokenizer (使用DistilBERT的tokenizer)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_seq_len = 77  # CLIP标准长度

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 图像处理
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.image_transform(image)
        
        # 文本处理
        text = self.texts[idx]
        text_input = self.tokenizer(
            text, 
            return_tensors='pt', 
            max_length=self.max_seq_len, 
            padding='max_length', 
            truncation=True
        )
        
        return {
            'image': image,
            'input_ids': text_input['input_ids'].squeeze(0),
            'attention_mask': text_input['attention_mask'].squeeze(0)
        }

# 示例数据
sample_data = {
    "image1.jpg": "a cat sitting on a couch",
    "image2.jpg": "a dog playing in the park",
    "image3.jpg": "a sunset over mountains"
}
dataset = CLIPDataset("path/to/images", sample_data)

# ----------------------
# 2. 模型构建
# ----------------------
class ImageEncoder(nn.Module):
    """简化版图像编码器 (使用ResNet18)"""
    def __init__(self, embed_size=512):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        
    def forward(self, x):
        return F.normalize(self.cnn(x), p=2, dim=-1)

class TextEncoder(nn.Module):
    """简化版文本编码器 (使用DistilBERT)"""
    def __init__(self, embed_size=512):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.proj = nn.Linear(self.bert.config.hidden_size, embed_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # 取[CLS] token
        return F.normalize(self.proj(pooled), p=2, dim=-1)

class CLIPModel(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_size)
        self.text_encoder = TextEncoder(embed_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # 可学习的温度参数
        
    def forward(self, batch):
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        return image_features, text_features, self.logit_scale.exp()

# ----------------------
# 3. 对比损失函数
# ----------------------
def clip_loss(image_features, text_features, logit_scale):
    # 计算相似度矩阵
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    
    # 对比损失 (对称的交叉熵)
    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2

# ----------------------
# 4. 训练循环
# ----------------------
def train_loop(model, dataloader, optimizer, device='cuda', epochs=10):
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            # 数据移至设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            image_features, text_features, logit_scale = model(batch)
            
            # 计算损失
            loss = clip_loss(image_features, text_features, logit_scale)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ----------------------
# 主函数
# ----------------------
if __name__ == "__main__":
    # 初始化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel(embed_size=512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 训练
    train_loop(model, dataloader, optimizer, device=device, epochs=10)
```
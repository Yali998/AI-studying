# ViT
## overview
vision transformer<br>
本质是一个图像编码器 (image encoder)，可以用于所有需要将图像作为输入数据的场景，如目标检测、语义分割、实例分割、图像生成、视频理解等

- 2020 google提出
- 和Bert几乎一致，encoder-only，同时不添加任何卷积结构的图像分类模型
- 证明了可以用统一的模型，来处理不同领域（语言/图像/视频）的任务

reference:
稀土掘金：[CV大模型系列之：全面解读VIT，它到底给植树人挖了多少坑](https://juejin.cn/post/7254341178258489404)

## 模型架构
![alt text](image.png)

### 公式
#### embedding后的输入，第0层：<br>

$z^0 = [x_{class}; x_1^E;...;x_N^E] + E_{pos}$<br>

#### LayerNorm -> MSA (multi-head self attention) -> residual:<br>

$z'_{\ell} = MSA\!\left(LN(z_{\ell-1})\right) + z_{\ell-1}
$

#### LayerNorm -> MLP -> residual:<br>

$z^\ell = MLP(LN(z'_{\ell})) + z'_{\ell}$

#### 取[CLS] token的输出:

$y = LN(z_0^L)$

## BERT (transformer) v.s ViT
### input的数据不同
图像数据转换成transformer能处理的序列数据<br>

#### 图像分成patches<br>
- original image: $H \times W \times C $
- patch: $P \times P$, $x_p \in R^{N \times (P^2 \cdot C)}$
- $N = \frac{HW}{P^2}$

*eg*: 256\*256\*3 -> 16\*16\*3 (共有196个patches)<br>

#### embedding<br>

$z_i = \text{PatchEmbed}(x_i)+\text{PosEmbed}(i)$<br>

**patch embedding (linear embedding)**<br>
把这个patch展开成一维向量(长度：16\*16\*3=768),然后乘以一个可学习的线性投影矩阵<br>

$\text{PatchEmbed}(x_i)=x_iE$

- $x_i \in R^{(P^2C)}$ 展开的patch向量
- $E \in R^{(P^2C) \times D}$ 可学习权重矩阵
- 输出：patch embedding， 维度 = $D$<br>

patch embedding的序列：$(x_1^E,x_2^E,...,x_N^E)$<br>

patch embedding的shape：$(N \times D)$

**classification token embedding** `[class]`<br>
类似于NLP的BERT里的`[CLS]` token，代表整个序列的全局信息。<br>
ViT中，在patch序列前面加一个learnable token，transformer编码后，这个token的输出向量被拿来做分类，判断原始图片中物体的类别。

**positional embedding**<br>
指示对应patch在原始图片中的位置。和Bert一样，这个位置向量是learnable的。

$E_{pos} \in R^{(N+1) \times D}$
- N+1 是因为加了一个`[class]`

### norm的位置不一样
**post-norm**: transformer里normalization在attention之后<br>
**pre-norm**: ViT里normalization在attention之前<br>

<font color = red>为什么视觉任务要将Norm层置于self-attention之前？ </font><br>
图像数据通常具有较大的尺度变化，Norm置于attention之前可以更有效地调整输入特征地尺度，数值分布更稳定，梯度容易传播。

## application
- CLIP
- BLIP
- GPT4o
- Qwen
- ...

## implementation
reference:

[Vision Transformer（ViT）PyTorch代码全解析](https://blog.csdn.net/weixin_44966641/article/details/118733341)

simplified version:
[vit_code](vit_code.py)
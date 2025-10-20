# Bert
paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)<br>
code: [bert](https://github.com/google-research/bert)<br>
reference resouce:<br>
- [69 BERT预训练【动手学深度学习v2】](https://www.bilibili.com/video/BV1yU4y1E7Ns/?spm_id_from=333.1391.0.0&p=4&vd_source=c40614f29fe4e0bd8bf156e97f9b3287)
- [BERT 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1PL411M7eQ/?spm_id_from=333.1391.0.0&vd_source=c40614f29fe4e0bd8bf156e97f9b3287)
- [Language Understanding with BERT](https://cameronrwolfe.substack.com/p/language-understanding-with-bert)

## overview
思路来源于bidirectional + transformer
- ELMo: bidirectional + RNN
- GPT: transformer架构但只能处理单向信息

架构：只有encoder的transformer<br>

### two version:
|name | blocks | hidden size | heads | params |
|----|----|---|---|---|
|Base|12|768|12|110M|
|Large|24|1024|16|340M|

数据集： >3B 词 <br>

### bert as a box
self-supervised learning<br>
input: 自然语言<br>
bert: 黑盒<br>
output: 对应的语言表征（向量）

![alt text](image.png)

拿到语言表征后，通过FC以适应下游任务，如接FC将向量映射到分类任务的维度。

![alt text](image-3.png)

### input
#### 对于Input的修改
1. add segment embedding
2. learnable positional embedding
3. sentence pair as sample: 可以区分不同句子是否为上下文

#### input sequence process
![alt text](image-1.png)
1. tokenization: 将原句子分割为单词
2. inserting special tokens: `[CLS]`-start, `[SEP]`-end
3. embedding

**关于embedding**:
positional embedding: learnable<br>
segment embedding: 用于区分句子<br>
word embedding: token的embedding<br>
![alt text](image-2.png)

## two pre-training tasks
两者是并行训练的<br>
![alt text](image-7.png)

### masked language model (MLM)
![alt text](image-4.png)

类似于完型填空，MLM每次随机(15%)将一些词元换乘`<mask>`<br>
这15%的词进一步：<br>
80% -> mask -> 学上下文<br>
10% -> 随机替换 -> 学鲁棒性<br>
10% -> 原词 -> 学真实语义 <br>
该策略让模型训练既接近真实输入，又能处理噪声，解决了预训练与微调真实词不匹配的问题。

**预训练-微调不匹配 pretrain-finetune mismatch**<br>
预训练阶段主要学习`<mask>`位置的预测任务，可能会过度依赖`<mask>`作为提示
- pretrain: MLM中，输入句子有mask token，用来让模型预测被遮住的词
- fine-tuning: 下游任务中，如分类or回答，输入中不会有mask，模型需要直接处理真实的句子

**MLM: loss**<br>
MLM的训练目标：<br>
- 只对被选中的15%位置计算预测误差(CE loss)
- 未被选中的词不会参与损失计算

$\mathcal{L}_{\text{MLM}} = - \sum_{i \in \mathcal{M}} \log P\big(x_i \,\big|\, x_{\backslash \mathcal{M}}\big)$
- $\mathcal{M}$: 被选中进行预测的token集合（15%）
- $x_i$: 第i个token
- $x_{\backslash \mathcal{M}}$: 除$\mathcal{M}$外的上下文token
- $P(x_i|x_{\backslash \mathcal{M}})$: 模型预测token $x_i$的概率

### next sentence prediction (NSP)
预测一个句子对中两个句子是不是相邻<br>
训练样本中：<br>
50%概率选择相邻句子对<br>
50%概率选择随机句子对<br>

![alt text](image-5.png)

## fine-tuning
![alt text](image-6.png)

1. same architecture as pre-trained model (token embedding, attention, feed-forward), and add output layer according to specific tasks
    - classification (semantic analysis): [CLS] token上接FC+softmax
    - 序列标注任务：在每个token的输出向量上接分类层，输出每个token的标签
    - Q & A: 输出start和end token的概率分布
2. initialize with pre-trained params
2. fine-tuning `all params + new layer` with downstream tasks dataset

## codes
unread
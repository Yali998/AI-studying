# Normalization
[归一化技术：从BatchNorm到RMSNorm——深度学习的稳定之锚](https://blog.csdn.net/qq_43664407/article/details/148357040)

why normalization?

在深度神经网络中，尤其是Transformer等模型，随着网络层数的加深，中间层的输入分布会发生剧烈变化，这被称为**内部协变量偏移**。这种不稳定性会使得训练过程变得困难，需要更小的学习率和更仔细的参数初始化。

归一化技术（如Batch Norm、Layer Norm）就是为了解决这个问题而生的。它们通过将输入数据重新调整为均值为0、方差为1的分布，来稳定网络的训练过程，从而允许使用更大的学习率，加速收敛，并一定程度上有正则化的效果。

## batch norm
在一个batch内，针对数据在通道尺度上计算均值和方差。

image (b, c, h, w)  # batch_size, channels, height, width <br>
batchNorm会将同batch、同channel的数据归一化为均值为0，方差为1的正态分布。

### formula
mean: $\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$<br>

variance: $\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$<br>

$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$

$y_i = \gamma \hat{x}_i + \beta$

### pros and cons
#### pros:
- 加速模型训练收敛：BatchNorm通过归一化手段，将每层输入强行拉回均值为0、方差为1的标准正态分布，
使激活函数的输入值分布在其梯度敏感区域，有效避免了梯度消失问题，显著加快了训练速度。
一般而言，BatchNorm常放置在非线性激活函数之前，有助于使输入到激活函数的值分布更加稳定。
- 提升模型泛化能力：不同批次数据的均值和方差存在差异，这种差异对于整体训练过程而言，类似于一种噪声。
不过，由于各批次间均值和方差的差异并不显著，能够保证该噪声具备一定的随机性且相对稳定（这也是训练过程中不直接使用整个训练集均值和方差的原因）。
这种噪声的引入，增强了模型的鲁棒性，使其不易对训练数据产生过度拟合，从而提升了模型在未知数据上的泛化能力。
#### cons:
- BatchNorm对Batch的大小依赖性较强。当Batch值较小时，计算得到的均值和方差稳定性较差
- 不适用于NLP领域

### application
主要用于CV中图像的处理，如使用BatchNorm对卷积层输出进行归一化处理。
- CNN
## layer norm
针对每个样本的特征维度进行归一化。
LayerNorm的归一化操作仅针对单个样本，不依赖于整个批次的数据，因此不受mini-batch大小的影响。

input: tensor (batch_size, sequence_length, hidden_size)

### formula
mean: $\mu = \frac{1}{H} \sum_{i=1}^{H} x_i $

variance: $\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 $

$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $

$y_i = \gamma \hat{x}_i + \beta $

- $ \mu$: 样本内特征维度的均值
- $\sigma^2$: 样本内特征维度的方差

### pros and cons
#### pros
- 提升泛化能力和收敛速度
- 对batch_size无要求：在小批量数据甚至单样本输入的场景下也能表现出色

#### cons
- 在CV领域相对劣势：CV任务中，数据通常以图像形式存在，具有明确的空间结构和通道信息。
BatchNorm能充分利用同批次图像在通道维度上的统计信息，而LayerNorm仅针对单个样本的特征为度进行归一化，会破坏不同样本间同通道特征的可比性。
- 改变特征向量关系：会改变不同样本间特征向量的相对大小关系。在某些对样本间特征向量相对关系敏感的任务中，这种改变可能会对模型性能产生一定的负面影响。

### application
- transformer
- 小 batch size
- RNN/LSTM

## RMSNorm
layerNorm中，减去均值的操作，只使用方差部分进行重新缩放，忽略均值中心化，模型的表现几乎不受影响。

将向量单位化，使其长度为1

### formula
$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$

$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma $

即：$y_i = \frac{x_i}{\text{RMS}(x_i)} \odot \gamma_i$

### pros
- 计算效率高：省略了均值计算
- 参数更少：省略了平移参数$\beta$

### application
- llama
- gpt-3
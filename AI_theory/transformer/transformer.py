# pytorch transformer implementation
'''
reference: https://blog.csdn.net/BXD1314/article/details/126187598
'''

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device  = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100

# training dataset
sentences = [
    # enc_input               dec_input                   target_output
    ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
    ['我 有 零 个 女 朋 友 P', 'S I have zero girl friend .', 'I have zero girl friend . E'],
    ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E']
]

# build vocabulary
src_vocab = {
    'P':0, '我':1, '有':2, '一':3, '个':4, '好':5, '朋':6, '友':7, '零':8, '女':9, '男':10
}
src_idx2word = {i:w for i,w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab) # 11

tgt_vocab = {
    'S':0, 'I':1, 'have':2, 'a':3, 'good':4, 'friend':5, '.':6, 'zero':7, 'girl':8, 'boy':9, 'E':10, '.':11
}
tgt_idx2word = {i:w for i,w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab) # 12

src_len = 8 # enc_input max length
tgt_len = 7 # dec_input = dec_output, max length

# transformer parameters
d_model = 512    # token embedding dimension and position embedding dimension
d_ff = 2048      # feed forward dimension
d_k = d_v = 64   # dimension of K(=Q), V
n_layers = 6     # number of encoder and decoder layers
n_heads = 8      # number of heads in multi-head attention

#------------------------------- data preprocessing -------------------------------#
def make_data(sentences):
    '''word sequence to index sequence'''
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]  # shape:[3, 8]

    def __getitem__(self, idx):
        '''return one sample(enc_input, dec_input, dec_output)'''
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
    
loader = Data.DataLoader(
    MyDataSet(enc_inputs, dec_inputs, dec_outputs),
    2,
    True,
)

#-------------------------------transformer model -------------------------------#
'''
- positional encoding
- multi-head attention
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1) # shape: [max_len, 1]，这样每一行是一个Pos
        '''e(2i * (-ln(10000)/d_model)) = e^(ln(1/10000^(2i/d_model))) = 1/10000^(2i/d_model)'''
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model
        ))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        '''
        pe shape: [max_len, d_model]
        unsqueeze(0) -> [1, max_len, d_model]
        transpose(0, 1) -> [max_len, 1, d_model]
        '''
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model] 处理后可以直接和 token embedding 相加
        self.register_buffer('pe', pe) # 注册为模型的固定属性张量，非模型参数，但需要保存下来
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
def get_attn_pad_mask(seq_q, seq_k):
    '''mask the padding token in sequences'''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # shape: [batch_size, 1, len_k], one is PAD token
    return pad_attn_mask.expand(batch_size, len_q, len_k) # shape: [batch_size, len_q, len_k]

def get_attn_subsequent_mask(seq):
    '''返回一个掩码矩阵，0代表可以访问，1代表被mask掉'''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # [batch_size, seq_len, seq_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # upper triangular matrix of 1, masked
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


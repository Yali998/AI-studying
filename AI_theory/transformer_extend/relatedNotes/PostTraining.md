# Post-training

让模型学会“更像人”“更有用”“更安全”。
包括： 
- Supervised Finetuning（SFT） 
- Preference-based methods（如 RLHF、DPO、Rejection Sampling 等）

reference:<br>
[人工智能LLM模型：奖励模型的训练、PPO 强化学习的训练、RLHF](https://blog.csdn.net/sinat_39620217/article/details/131776129?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522986f25fe35fa20713a63ad1fd0d50c87%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=986f25fe35fa20713a63ad1fd0d50c87&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~top_positive~default-1-131776129-null-null.nonecase&utm_term=%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B&spm=1018.2226.3001.4450)
## SFT: supervised finetuning
用人类编写的高质量问答样本直接训练模型，让它学会“好回答”。

data style:
```
prompt -> response
```

training goal:<br>
minimise the loss of prediction and target: cross entropy

⚠️ 缺点： 
- 模型只学到模仿训练数据； 
- 无法学习“偏好”或“奖励信号”； 
- 一旦遇到模糊、未标注的数据，不知道哪种回答更好。

👉 所以我们需要下一阶段：偏好学习（Preference Learning）。

## reject sampling
基于拒绝采样的偏好训练：用于在多个可能回答中选择“最好”的那一个。
是RLHF的轻量化替代方案。

1. 使用SFT模型对同一个prompt生成多个回答：${y_1, y_2, ..., y_n}$。
2. 人类or自动模型评估这些回答，选出最好的一个。
3. 只用这些“最佳答案”再次SFT模型，形成新模型。

## RLHF: reinforcement learning from human feedback
基于人类反馈的强化学习。
```
Pretraining → SFT → Reward Model → RLHF (PPO)
  |             |        |            |
  |             |        |            └── Fine-tune to align with reward
  |             |        └── Learn human preference
  |             └── Learn human-written answers
  └── Learn language patterns
```
### stage1: SFT

### stage2: RM(reward model)
1. 人类评审者对模型生成的多个候选答案进行排序
```A > B > C > D```
2. 训练reward model, 学习该偏好排序
3. RM:
   4. input: (prompt, answer)
   5. ouput: score

prompt和answer的匹配度越高，则奖励模型输出的分数也越高。

loss function: pairwise ranking loss

### stage3: reinforcement learning
用强化学习调整语言模型，当模型在生成回答时：
- reward model给出奖励
- PPO优化使模型输出能最大化奖励 $\text{maximise}~\mathbb{E}[\mathbf{R}_\theta(\text{prompt}, \text{answer})]$
- 加入KL惩罚项，防止模型偏离原始语言能力太远 $\mathbf{L} = - \mathbf{R}_\theta + \beta \cdot \mathbf{D}_{\text{KL}}(\pi_\theta||\pi_{\text{SFT}})$

#### PPO: proximal policy optimization
近端策略优化：对设定的目标函数通过随机梯度下降进行优化。

#### DPO: direct preference optimization
直接偏好优化：不再显式训练奖励模型或使用强化学习，而是直接利用人类偏好数据（preferred / dispreferred responses）  
对语言模型参数 $(\pi_\theta)$ 进行优化。

preference pair: $(x, y^+, y^-)$

$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y^+, y^-)}[\log\sigma(\beta(\log\frac{\pi_\theta(y^+|x)}{\pi_{ref}(y^+|x)} - \log\frac{\pi_\theta(y^-|x)}{\pi_{ref}(y^-|x)})]$
- $\pi_\theta$: the model
- $\pi_{ref}$: reference model (sft model)
- $\sigma(\cdot)$: sigmoid
- $\beta$: temperature coefficient

模型参数更新方向：让生成$y^+$的每个词的概率更大，而生成$y^-$的每个词的概率更小。

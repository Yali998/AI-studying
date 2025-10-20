- prompt
- special dataset -> sft
- RAG: retrieval and generation

底层知识库做向量化

数据准备阶段
- 数据提取 -> chunk文本分割 -> 向量化embedding -> 数据入库
  - 按照句号做分割
  - 固定长度分割（但会损失很多语义信息）
  - 语义分割（有相关模型）

应用阶段
- 用户提问 -> 数据检索（召回）->


向量化：如bert, clip, BGE

向量数据库包括： FAISS, Chromadb, ES, milvus等

检索：用用户提问去检索底层知识库最相关的内容<br>
query -query embedding - doc embedding
- 向量相似度：余弦相似度、欧氏距离、曼哈顿距离等
- 全文检索/关键词构建倒排索引：关键词匹配

RAG中的embedding模型很重要，承担者query理解、知识库向量化构建

query, good, bad: <br>
带margin的triple loss: (query-good) - (query-bad) + margin
双塔 embedding  (query - doc)

### 联网检索
搜索关键词，给搜索引擎

## RAG 和 大模型微调
- prompt工程：激发大模型本身的能力 
- 训练是知识输入
- RAG是知识外挂


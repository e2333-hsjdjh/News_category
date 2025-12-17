# News Category Classifier (Transformer + Stacking Ensemble)

这是一个基于 PyTorch 实现的新闻文本分类项目。该项目从零构建了一个基于 **Transformer Encoder** 的轻量级分类器，并采用了 **Stacking 集成学习** 策略，通过并行训练两个基础模型（标题模型和摘要模型），再利用一个元模型（Meta Model）来融合它们的预测结果，从而实现高精度的分类。

## 📂 项目结构

```
.
├── NewsClassifier.ipynb    # 核心笔记本：包含所有训练、评估和推理代码
├── best_model_headline.pt  # 训练好的标题模型权重
├── best_model_description.pt # 训练好的摘要模型权重
├── best_meta_model.pt      # 训练好的元模型权重
├── DATA/
│   ├── dataset.json        # 原始数据集
│   └── ...
└── README.md               # 项目说明文档
```

## 数据库：

本项目运用的数据库是kaggle网站中的 News Category Dataset，原始数据为文件DATA/News_Category_Dataset.json

## 训练前的数据处理：

由于模型的大小，将原数据中的种类进行了适当的合并具体为：

| 原始类别 | 新分类 |
| :--- | :--- |
| ARTS / COMEDY / STYLE | Arts & Entertainment |
| BLACK VOICES / WOMEN | Society & Identity |
| POLITICS / WORLD NEWS | Politics & World Affairs |
| BUSINESS / MONEY | Business & Economy |
| SCIENCE / TECH | Science, Tech & Environment |
| WELLNESS | Health & Wellness |
| FOOD & DRINK / TRAVEL | Lifestyle & Leisure |
| EDUCATION / COLLEGE | Education & Youth |

## 程序可复现操作：

为了确保实验结果的可复现性，请按照以下步骤操作：

1.  **安装依赖库**:
    确保您的 Python 环境中安装了 `requirements.txt` 中列出的特定版本库。
    ```bash
    pip install -r requirements.txt
    ```
2.  **随机种子设置**:
    代码中已默认设置随机种子为 `42` (`set_seed(42)`)。请勿更改此设置，以保证模型初始化和数据划分的一致性。

## 🚀 程序执行流程详解

本项目的 `NewsClassifier.ipynb` 笔记本按照以下逻辑步骤执行：

### 1. 环境准备与数据加载
*   **环境配置**: 设置随机种子以保证结果可复现；自动检测计算设备，优先使用 Apple Silicon 的 **MPS** 加速，其次是 CUDA，最后回退到 CPU。
*   **数据读取**: 加载 `DATA/dataset.json`，并统计各类别的样本分布，以便后续处理类别不平衡问题。

### 2. 数据预处理 (Preprocessing)
*   **自定义分词器 (`WhitespaceTokenizer`)**:
    *   实现了一个轻量级的分词器，不依赖 BERT 等预训练模型。
    *   功能包括：转小写、正则切分标点符号、构建词表、将文本转为 ID 序列。
    *   特殊 Token: `<pad>` (填充), `<unk>` (未知词), `<cls>` (分类标识)。
*   **数据集构建 (`NewsDataset`)**:
    *   继承自 `torch.utils.data.Dataset`。
    *   负责将文本和标签转换为 PyTorch Tensor，并生成 Attention Mask。

### 3. 模型架构 (Model Architecture)
我们构建了一个基于 Transformer 的文本分类器 `NewsClassifier`：
*   **Embedding 层**: 将离散的 Token ID 映射为稠密向量。
*   **Positional Encoding**: 由于 Transformer 是并行处理的，需要注入位置编码（正弦/余弦函数）来保留词序信息。
*   **Transformer Encoder**: 核心组件，利用多头自注意力机制 (Multi-Head Self-Attention) 捕捉文本中的长距离依赖和语义特征。
    *   **编码层 (Encoder Layers)**: 2 层。
    *   **解码层 (Decoder Layers)**: 0 层 (本任务为文本分类，不需要生成式解码器)。
*   **Classifier Head**: 简单的全连接层，将编码后的语义向量映射到类别概率。

#### Transformer 训练与前向传播流程详解
在训练过程中，数据在模型中的流转步骤如下：
1.  **输入处理**: 文本被转换为 Token ID 序列，并生成 \`Attention Mask\`（用于标记有效词和填充词）。
2.  **向量化 (Embedding + Positional)**: Token ID 转换为词向量，并叠加位置编码，使模型理解词序。
3.  **自注意力机制 (Self-Attention)**:
    *   **Q, K, V 计算**: 输入向量分别映射为 Query (查询), Key (键), Value (值) 矩阵。
    *   **Attention Score**: 计算 Q 和 K 的相似度，衡量词与词之间的关联。
    *   **Masking**: 利用 `Attention Mask` 屏蔽掉 `<pad>` 填充位置，确保模型只关注真实内容。
    *   **聚合**: 根据关联度加权聚合 V (Value)，生成包含深层语义的上下文向量。
4.  **前馈与残差 (FeedForward & AddNorm)**: 通过全连接层提取更高阶特征，并利用残差连接防止梯度消失。
5.  **分类输出**: 提取 \`<cls>\` 标记位置的向量作为整句表示，通过线性层输出各类别的 Logits。
6.  **损失计算**: 使用 \`CrossEntropyLoss\` 计算预测分布与真实标签的差异，并通过反向传播更新参数。

### 4. 并行双流训练 (Parallel Dual-Stream Training)
为了充分利用新闻的“标题”和“摘要”信息，我们训练了两个独立的模型。为了提高效率，使用了 **多线程并行 (Multi-threading)** 技术：
*   **Headline Model**: 仅使用新闻标题进行训练。
*   **Description Model**: 仅使用新闻摘要进行训练。
*   **并行执行**: 使用 `concurrent.futures.ThreadPoolExecutor` 同时启动两个训练任务，显著缩短了总训练时间。
*   **优化策略**:
    *   **类别权重 (Class Weights)**: 针对数据不平衡，计算了 Loss 权重。
    *   **学习率调度 (Scheduler)**: 使用 `ReduceLROnPlateau`，当验证集准确率不再提升时自动降低学习率。

### 5. 集成学习：元模型训练 (Meta-Learning / Stacking)
这是本项目提升效果的关键步骤。我们不仅仅是简单地平均两个模型的预测，而是训练了一个 **Meta Model** 来学习如何组合它们。
*   **数据生成**: 使用验证集数据，分别通过训练好的 Headline Model 和 Description Model，提取它们的预测概率分布（Softmax 输出）。
*   **特征拼接**: 将两个模型的预测概率拼接成一个长向量，作为 Meta Model 的输入。
*   **Meta Classifier**: 一个简单的多层感知机 (MLP)，学习从这些概率特征到真实标签的映射。它能自动学会何时该信赖标题模型，何时该信赖摘要模型。

### 6. 智能推理 (Inference)
`EnsemblePredictor` 类封装了最终的推理逻辑，支持自适应处理缺失数据：
1.  **完整输入**: 若同时提供标题和摘要，分别获取两个基础模型的预测概率，输入 Meta Model 得到最终结果。
2.  **缺失摘要**: 若只有标题，直接使用 Headline Model 的预测结果。
3.  **缺失标题**: 若只有摘要，直接使用 Description Model 的预测结果。

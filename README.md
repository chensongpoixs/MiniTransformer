# MiniTransformer

从零实现的 Transformer 模型，用于英译中机器翻译任务。

## 项目结构

```
MiniTransformer/
├── train.py                 # 训练主入口
├── eval.py                  # 评估脚本（空）
├── infer.py                 # 推理脚本（空）
├── main.py                  # PyCharm 默认模板（无用）
├── requirements.txt         # 依赖文件（空）
├── README.md                # 项目说明
├── AGENTS.md                # Agent 指引
├── .gitignore
│
├── models/                  # Transformer 模型定义
│   ├── __init__.py          # 导出所有模型组件
│   ├── transformer.py       # Transformer 顶层封装
│   ├── encoder.py           # 编码器（N × EncoderLayer）
│   ├── decoder.py           # 解码器（N × DecoderLayer）
│   ├── attention.py         # 多头注意力机制
│   ├── embedding.py         # Token 嵌入层
│   ├── positional_encoding.py # 正弦位置编码
│   ├── feed_forward.py      # 前馈网络
│   ├── generator.py         # 输出层（线性 + softmax）
│   ├── layer_norm.py        # 自定义 LayerNorm
│   └── utils.py             # 工具函数（clones, SublayerConnection）
│
├── datasets/                # 数据处理
│   ├── __init__.py          # 导出 MTDataset, beam_search
│   ├── mt_dataset.py        # 机器翻译数据集类
│   ├── beam_decoder.py      # Beam Search 解码实现
│   ├── preprocess.py        # 数据预处理（空）
│   └── tokenizer/
│       └── tokenizer.py     # SentencePiece 分词器加载
│
├── trainer/                 # 训练相关
│   ├── __init__.py          # 导出 make_model, evaluate
│   ├── trainer.py           # 模型构建（make_model）
│   ├── evaluator.py         # BLEU 评估（sacrebleu）
│   ├── checkpoint.py        # 检查点保存（空）
│   └── metrics.py           # 指标计算（空）
│
├── optimizer/               # 优化器
│   ├── __init__.py
│   ├── adam.py              # NoamOpt + MultiGPULossCompute
│   ├── scheduler.py         # 学习率调度（空）
│   └── warmup.py            # 预热策略（空）
│
├── utils/                   # 工具模块
│   ├── __init__.py          # 导出 create_exp_folder
│   ├── config.py            # 超参数配置（硬编码）
│   ├── visualization.py     # 实验目录创建
│   ├── seed.py              # 随机种子（空）
│   ├── logger.py            # 日志（空）
│   └── distributed.py       # 分布式训练（空）
│
├── inference/               # 推理模块（均为空）
│   ├── beam_search.py
│   ├── greedy_search.py
│   └── predict.py
│
├── losses/                  # 损失函数（均为空）
│   ├── loss.py
│   └── label_smoothing.py
│
├── tokenizer/               # 预训练分词器
│   ├── eng.model            # 英文 SentencePiece 模型
│   ├── eng.vocab            # 英文词表
│   ├── chn.model            # 中文 SentencePiece 模型
│   ├── chn.vocab            # 中文词表
│   ├── tokenize.py          # 分词器训练脚本
│   └── tokenize.ipynb       # 分词器训练 Notebook
│
├── data/                    # 数据文件
│   ├── corpus.en            # 英文语料
│   ├── corpus.ch            # 中文语料
│   ├── get_corpus.py        # 从 JSON 提取语料
│   ├── analyze_corpus.py    # 语料分析工具
│   ├── json_unicode_preview.py
│   └── json/
│       ├── train.json       # 训练集（英文, 中文）
│       ├── dev.json         # 验证集
│       └── test.json        # 测试集
│
├── configs/                 # 配置文件（均为空）
│   ├── train.yaml
│   ├── model.yaml
│   └── inference.yaml
│
├── weights/                 # 预训练权重
│   ├── best_bleu_17.11.pth
│   └── transformer_model.zip
│
├── run/                     # 训练输出目录
│   └── train/
│       └── expN/weights/    # 每次实验的权重
│
├── checkpoints/             # 检查点目录（空）
├── logs/                    # 日志目录
│   └── tensorboard/
└── scripts/                 # 脚本目录（空）
```

## 模型架构

```
┌─────────────────────────────────────────────────────┐
│                  Transformer                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  输入序列 ──→ [Embedding + PositionalEncoding]       │
│                    │                                │
│                    ▼                                │
│           ┌─────────────────┐                      │
│           │    Encoder      │  × N=6               │
│           │  ┌───────────┐  │                      │
│           │  │ Self-Attn │──┼──→ LayerNorm         │
│           │  └───────────┘  │      │               │
│           │  ┌───────────┐  │      │               │
│           │  │    FFN    │──┼──→ LayerNorm         │
│           │  └───────────┘  │                      │
│           └─────────────────┘                      │
│                    │                                │
│                    ▼ memory                         │
│           ┌─────────────────┐                      │
│           │    Decoder      │  × N=6               │
│           │  ┌───────────┐  │                      │
│           │  │ Self-Attn │──┼──→ LayerNorm         │
│           │  └───────────┘  │      │               │
│           │  ┌───────────┐  │      │               │
│           │  │ Cross-Attn│──┼──→ LayerNorm         │
│           │  │ (memory)  │  │      │               │
│           │  └───────────┘  │      │               │
│           │  ┌───────────┐  │      │               │
│           │  │    FFN    │──┼──→ LayerNorm         │
│           │  └───────────┘  │                      │
│           └─────────────────┘                      │
│                    │                                │
│                    ▼                                │
│           [Generator (Linear → Softmax)]           │
│                    │                                │
│                    ▼                                │
│               输出序列                               │
└─────────────────────────────────────────────────────┘
```

## 超参数配置

所有超参数在 `utils/config.py` 中硬编码：

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 512 | 模型维度 |
| d_ff | 2048 | 前馈网络隐藏层维度 |
| n_heads | 8 | 注意力头数 |
| n_layers | 6 | 编码器/解码器层数 |
| d_k / d_v | 64 / 64 | 每个头的维度 |
| dropout | 0.1 | 丢弃率 |
| src_vocab_size | 32000 | 源语言词表大小 |
| tgt_vocab_size | 32000 | 目标语言词表大小 |
| batch_size | 32 | 批次大小 |
| epoch_num | 300 | 训练轮数 |
| lr | 3e-4 | 学习率 |
| beam_size | 3 | Beam Search 宽度 |
| max_len | 60 | 最大解码长度 |

特殊 token：
- PAD = 0, UNK = 1, BOS = 2, EOS = 3

## 数据格式

训练/验证/测试数据为 JSON 文件，每条记录为 `[英文句子, 中文句子]`：

```json
[
  ["Hello, how are you?", "你好，你怎么样？"],
  ["I love programming.", "我喜欢编程。"]
]
```

## 运行环境

项目在 conda 的 `deeplearn` 环境中运行。

### 环境激活

```bash
conda activate deeplearn
```

### 依赖安装

```bash
pip install torch sentencepiece sacrebleu tqdm numpy
```

## 使用方法

### 1. 训练

```bash
python train.py
```

- 自动使用 GPU 0（`CUDA_VISIBLE_DEVICES=0`）
- 模型权重保存到 `run/train/expN/weights/`
- 每轮训练后计算 BLEU 分数，保存最优模型

### 2. 训练分词器（如需重新训练）

```bash
cd tokenizer
python tokenize.py
```

### 3. 语料预处理

```bash
cd data
python get_corpus.py          # 从 JSON 提取语料
python analyze_corpus.py      # 分析语料统计信息
```

## 依赖

```
torch
sentencepiece
sacrebleu
tqdm
numpy
```

## 已知问题

1. **`train.py` 缺少 `import os`** — 第 75-76 行调用 `os.path.exists` / `os.remove` 但未导入 `os`，替换旧模型时会崩溃
2. **`datasets/beam_decoder.py` 使用非相对导入** — `from mt_dataset import ...` 仅在 CWD 为 `datasets/` 时有效
3. **`evaluator.py` 导入错误** — 从 `datasets.__init__` 导入 `beam_decoder`（模块名），导入时会失败
4. **`requirements.txt` 为空** — 依赖需手动安装
5. **`eval.py`、`infer.py` 及多个子模块为空** — 功能尚未实现

## 优化器

使用 Noam 学习率调度器（warmup = 10000 步）包裹 Adam：

```python
lr = factor × (d_model^(-0.5)) × min(step^(-0.5), step × warmup^(-1.5))
```

- β = (0.9, 0.98), ε = 1e-9

## 损失函数

```python
CrossEntropyLoss(ignore_index=0, reduction='sum')
```

- `ignore_index=0`：忽略 PAD token
- `reduction='sum'`：对有效 token 的损失求和

## 训练细节

- 使用 `nn.DataParallel` 进行多 GPU 训练
- 模型构建时通过 `.to(DEVICE)` 移至 GPU
- 每个 epoch 完成训练后进行验证，计算 BLEU 分数
- 保存两个模型：最优模型（best BLEU）和最后模型

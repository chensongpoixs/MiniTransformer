# AGENTS.md

## 项目简介

从零实现的 Transformer（PyTorch），用于**英译中机器翻译**。SentencePiece BPE 分词器、Beam Search 解码、SacreBLEU 评估。

## 运行环境

conda 环境：`deeplearn`

```bash
conda activate deeplearn
```

## 运行方式

```bash
python train.py
```

唯一入口。内部设置 `CUDA_VISIBLE_DEVICES=0`。训练数据 `data/json/train.json`，验证数据 `data/json/dev.json`。模型权重保存到 `run/train/expN/weights/`。

**无 lint、typecheck、test 命令。** `scripts/` 目录为空。

## 硬编码配置（非 YAML）

所有超参数在 `utils/config.py` — `configs/` 下的 YAML 文件为空。关键值：

| 参数 | 值 |
|------|-----|
| d_model / d_ff / n_heads / n_layers | 512 / 2048 / 8 / 6 |
| batch_size / epochs / lr | 32 / 300 / 3e-4 |
| vocab_size（源 & 目标） | 32000 |
| 特殊 token | PAD=0, UNK=1, BOS=2, EOS=3 |
| beam_size / max_len | 3 / 60 |

## 数据格式

`data/json/{train,dev,json}.json` — 每条为 `[英文句子, 中文句子]`。原始语料在 `data/corpus.{en,ch}`。

## 分词器

预训练 SentencePiece 模型在 `tokenizer/{eng,chn}.model`。通过 `datasets/tokenizer/tokenizer.py` 加载，使用**相对路径**（`./tokenizer/eng.model`），必须从项目根目录运行。

## 关键坑点

- **`train.py` 缺少 `import os`** — 第 75-76 行调用 `os.path.exists` / `os.remove` 但未导入 `os`，替换旧 best 模型时会崩溃。
- **`eval.py` 和 `infer.py` 为空** — 评估和推理脚本尚未实现。
- **`datasets/beam_decoder.py` 使用非相对导入**（`from mt_dataset import ...`）— 仅在 CWD 为 `datasets/` 时有效。
- **`evaluator.py` 从 `datasets.__init__` 导入 `beam_decoder` 和 `beam_search`**，但 `beam_decoder` 是模块名而非函数 — 导入时会失败。
- **`requirements.txt` 为空** — 依赖为：`torch`、`sentencepiece`、`sacrebleu`、`tqdm`、`numpy`。

## 架构

```
models/
  transformer.py   # 顶层：Encoder + Decoder + Embeddings + Generator
  encoder.py       # N× EncoderLayer（自注意力 + FFN）
  decoder.py       # N× DecoderLayer（自注意力 + 交叉注意力 + FFN）
  attention.py     # MultiHeadedAttention
  embedding.py     # Token 嵌入 + 缩放
  positional_encoding.py  # 正弦位置编码
  feed_forward.py  # 两层线性 + ReLU
  generator.py     # 最后一层线性 → softmax
  layer_norm.py    # 自定义 LayerNorm
  utils.py         # clones(), SublayerConnection

optimizer/adam.py  # NoamOpt（warmup 调度器）+ MultiGPULossCompute
trainer/trainer.py # make_model() — 构建完整 Transformer
trainer/evaluator.py  # sacrebleu BLEU 评估
```

## 代码约定

- 注释为中文 — 新增注释请保持此风格。
- 模型在构建时通过 `.to(DEVICE)` 移至 GPU；`DEVICE` 在导入时计算。
- 训练使用 `nn.DataParallel`，`device_id=[0]`。
- 损失函数：`CrossEntropyLoss(ignore_index=0, reduction='sum')`。
- 优化器：Noam 调度（warmup=10000 步）包裹 Adam(β=(0.9, 0.98), ε=1e-9)。

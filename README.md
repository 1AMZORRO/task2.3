# task2.2 — RNAFM RNA Fitness 零样本预测与监督训练

使用 [RNAFM](https://huggingface.co/multimolecule/rnafm) 预训练模型提取 RNA 序列 embedding，对多个 RNA 适应性（fitness）数据集进行**零样本（zero-shot）预测**和**监督回归训练**，最终输出 loss 曲线与 Spearman 分数曲线可视化。

---

## 数据集

`data/` 目录下包含 **30 个** RNA 适应性 DMS（Deep Mutational Scanning）数据集，覆盖核酶（ribozyme）、tRNA、mRNA、适配体（aptamer）等多种 RNA 类型。

每个 CSV 文件格式：

| 列名 | 说明 |
|------|------|
| `mutant` | 突变描述（空 = 野生型） |
| `DMS_score` | 实验测量的适应性得分 |
| `sequence` | RNA 序列（使用 AUGC 碱基） |

---

## 方法

### 1. Embedding 提取（冻结 RNAFM）

加载 `multimolecule/rnafm` 预训练权重，冻结全部参数，对每条序列提取 **mean-pooled** 最后一层 hidden state 作为固定维度表示（维度由所选模型的 `hidden_size` 决定）。

### 2. Zero-shot 预测（无需训练）

对**含野生型**的数据集，计算突变体 embedding 与野生型 embedding 的**余弦相似度**，直接与 DMS_score 计算 Spearman ρ。

可选：计算**伪对数似然（PLI）**得分：

$$\text{PLI}(\text{seq}) = \frac{1}{L} \sum_{i=1}^{L} \log P(x_i \mid x_{-i})$$

适用于无野生型的数据集，使用 `--use_pli` 参数启用。

### 3. 监督回归训练

在冻结的 embedding 之上训练一个小型回归头（Linear → ReLU → Dropout → Linear），使用 MSE 损失优化，按 `train_ratio`（默认 80/20）划分训练/验证集，每个 epoch 记录：
- 训练集 MSE loss
- 验证集 MSE loss
- 验证集 Spearman ρ

### 4. 可视化输出

| 文件 | 说明 |
|------|------|
| `results/curves/<数据集名>.png` | 双子图：左 loss 曲线，右 Spearman 曲线 |
| `results/summary_spearman.png` | 全数据集最终 Spearman ρ 汇总柱状图 |
| `results/embeddings/<数据集名>.npy` | 各数据集 embedding 矩阵 |
| `results/predictions/<数据集名>.csv` | 带预测分数的原始数据 |
| `results/results.csv` | 所有数据集指标汇总 |

---

## 环境配置

### 硬件要求

- GPU：NVIDIA RTX 3090（24 GB 显存）或其他 CUDA 11.8 兼容显卡
- CUDA：≥ 11.1（推荐 11.8）

### 方式一：Conda（推荐）

```bash
conda env create -f environment.yml
conda activate task2
```

### 方式二：pip

```bash
# 1. 安装 PyTorch（CUDA 11.8）
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装其余依赖
pip install -r requirements.txt
```

> **注意**：`requirements.txt` 中 `transformers==5.0.0` 为严格版本要求，  
> `multimolecule==0.0.9` 仅与此版本兼容，请勿随意升级。

---

## 运行

### 训练单个数据集（推荐）

```bash
# 只训练 Domingo_2018_tRNA 数据集（传入名称，无需 .csv 后缀）
python train.py \
    --dataset   Domingo_2018_tRNA \
    --data_dir  data/ \
    --output_dir results/ \
    --device    cuda \
    --epochs    50

# 或带上 .csv 后缀也可以
python train.py --dataset Domingo_2018_tRNA.csv --data_dir data/ --device cuda
```

### 训练全部数据集（不传 --dataset）

```bash
python train.py \
    --data_dir  data/ \
    --output_dir results/ \
    --device    cuda \
    --epochs    50 \
    --batch_size 64
```

### 快速验证（每数据集取前 200 条，20 轮）

```bash
python train.py \
    --dataset   Domingo_2018_tRNA \
    --data_dir  data/ \
    --output_dir results/ \
    --device    cuda \
    --max_seqs  200 \
    --epochs    20
```

### 启用 Zero-shot PLI 基线（较慢）

```bash
python train.py \
    --dataset   Domingo_2018_tRNA \
    --data_dir  data/ \
    --output_dir results/ \
    --device    cuda \
    --use_pli
```

### 全部参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `None` | 指定单个数据集名称（如 `Domingo_2018_tRNA`），不传则处理全部 |
| `--data_dir` | `data` | CSV 数据集目录 |
| `--output_dir` | `results` | 输出目录 |
| `--model_name` | `multimolecule/rnafm` | HuggingFace 模型名称或本地路径 |
| `--device` | `cuda` | 运行设备（`cuda` / `cpu`） |
| `--batch_size` | `64` | Embedding 提取批大小 |
| `--epochs` | `50` | 每数据集训练轮次 |
| `--lr` | `1e-3` | Adam 学习率 |
| `--train_ratio` | `0.8` | 训练集比例 |
| `--hidden_dim` | `64` | 回归头隐藏层维度（`0` = 纯线性） |
| `--use_pli` | `False` | 同时计算 zero-shot PLI 基线 |
| `--max_seqs` | `None` | 每数据集最大序列数（调试用） |
| `--seed` | `42` | 随机种子 |

---

## 输出示例

### Loss 曲线 + Spearman 曲线（每数据集）

每个数据集生成一张双子图，例如 `results/curves/Domingo_2018_tRNA.png`：

- **左图**：蓝色 = Train Loss，橙色虚线 = Val Loss
- **右图**：绿色 = Val Spearman ρ，紫色虚线 = Zero-shot 余弦基线（含野生型数据集）

### 汇总 Spearman 柱状图

`results/summary_spearman.png`：横向柱状图展示所有数据集的最终验证集 Spearman ρ，绿色为正相关，红色为负相关。

---

## 项目结构

```
task2.1/
├── data/                      # 30 个 RNA fitness DMS 数据集
│   ├── Domingo_2018_tRNA.csv
│   ├── Beck_2022_ribozyme.csv
│   └── ...（共 30 个）
├── train.py                   # 主训练脚本（embedding 提取 + 回归训练 + 可视化）
├── environment.yml            # Conda 环境（Python 3.10, PyTorch 2.1.0, CUDA 11.8）
├── requirements.txt           # pip 依赖（精确版本）
└── README.md
```

---

## 参考文献

- **RNAFM**：Model card: https://huggingface.co/multimolecule/rnafm
- **MultiMolecule**：https://multimolecule.danling.org

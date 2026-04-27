# RNA 零样本突变效应预测

利用预训练 RNA 语言模型 **RNA-FM**（通过 [multimolecule](https://multimolecule.danling.org/) 加载）对 RNA 突变效应进行**零样本预测**。

- **全冻结推理**：不解冻、不更新任何模型权重，直接使用预训练好的参数进行推理
- **零样本预测**：无需任何监督训练，直接提取模型输出的 logits 进行打分

---

## 打分方案

| 方案 | 全称 | 说明 |
|------|------|------|
| **WT-LLR** | Wild-Type conditioned Log-Likelihood Ratio | 对野生型序列进行一次前向传播，在各突变位点计算对数似然比：`Σ [log P(mut_i｜WT) − log P(wt_i｜WT)]`。正值表示模型认为突变碱基更符合上下文 |
| **PLL-D** | Pseudo-Log-Likelihood Difference | 对野生型和突变体序列各进行一次前向传播，计算全序列伪对数似然之差：`PLL(mut) − PLL(WT)`。正值表示突变体序列整体更符合模型分布 |

两种方案均为高效近似（无需 masking），推理速度快。

---

## 数据集

`data/` 目录包含 **30 个** RNA 深度突变扫描（DMS）数据集，涵盖多种 RNA 功能类型：

| 类型 | 数据集示例 |
|------|-----------|
| Ribozyme | Andreasson_2020, Beck_2022, Roberts_2023_* 等 |
| tRNA | Domingo_2018, Guy_2014, Li_2016 |
| mRNA | Julien_2016, Ke_2017 |
| Aptamer | Tome_2014_GFP, Tome_2014_NELFE |

每个 CSV 文件包含以下字段：

| 字段 | 说明 |
|------|------|
| `mutant` | 突变描述（如 `C10A,U56C`），空白行为野生型 |
| `DMS_score` | 实验测定的突变效应分数 |
| `sequence` | RNA 序列（使用 U 代替 T） |

> **注意**：使用 N-notation（如 `N24A,N25G`）的组合库数据集（Kobori、Peri、Townshend 系列）野生型碱基未知，将自动跳过。无显式野生型行的数据集（如 Beck、Julien、Pitt 等）会自动从突变注释还原野生型序列。

---

## 评价指标

| 指标 | 含义 |
|------|------|
| **Spearman (SR)** | 预测打分与实验 DMS score 的斯皮尔曼等级相关系数，衡量排序一致性 |
| **AUC** | 以中位数 DMS score 为阈值二值化后的 ROC 曲线下面积 |
| **MCC** | Matthews 相关系数，衡量二分类性能（正负类均衡敏感） |

---

## 环境配置

```bash
conda env create -f environment.yml
conda activate task2
```

---

## 运行

```bash
python train.py
```

程序会自动：
1. 从 HuggingFace Hub 下载并加载 `multimolecule/rnafm` 预训练权重（首次运行需联网）
2. 全冻结模型参数，仅进行推理
3. 对每个数据集分别用 WT-LLR 和 PLL-D 两种方案打分
4. 计算 Spearman、AUC、MCC 三项指标并打印汇总统计
5. 将结果保存至 `results/` 目录

---

## 输出文件

运行完成后，`results/` 目录包含：

| 文件 | 内容 |
|------|------|
| `metrics.csv` | 各数据集 WT-LLR 和 PLL-D 的详细指标（SR / AUC / MCC / n） |
| `summary_boxplot.png` | 箱型图：各指标跨数据集分布（WT-LLR vs PLL-D 对比） |
| `heatmap.png` | 热图：每个数据集在两种方案、三项指标上的完整性能矩阵 |
| `wt_llr_vs_pll_d_spearman.png` | 散点图：两种打分方案 Spearman 相关性的跨数据集对比 |

---

## 方法原理

RNA-FM 是基于 BERT 架构的 RNA 掩码语言模型，在大规模非编码 RNA 序列上预训练。

**零样本打分原理：**

- **WT-LLR**：模型的语言建模头在每个序列位置输出词表上的 logit 分布，反映"该位置应出现哪种碱基"的先验偏好。在突变位点，`log P(mut) − log P(wt)` 量化了突变的"语言模型合理性"，可直接用于零样本突变效应预测。

- **PLL-D**：对整条序列计算伪对数似然（所有位点 log P 之和），突变体与野生型的差值反映了整体序列适应性的变化。计算高效，无需逐位置 masking。

两种方案均**无需任何训练或微调**，直接利用预训练权重进行零样本推理。

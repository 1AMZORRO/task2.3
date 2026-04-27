#!/usr/bin/env python3
"""
RNA 零样本突变效应预测
======================
使用预训练 RNAFM 模型（全冻结参数）对 RNA 突变效应进行零样本预测。

打分方案：
  - WT-LLR : Wild-Type conditioned Log-Likelihood Ratio
              以野生型为条件的对数似然比
  - PLL-D  : Pseudo-Log-Likelihood Difference
              伪对数似然差

评价指标：
  - Spearman 相关系数（SR）
  - AUC（ROC 曲线下面积）
  - MCC（Matthews 相关系数）
"""

import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

# ── 优先使用 multimolecule 专用类；若不可用则退回 AutoModel ──────────
try:
    from multimolecule import RnaTokenizer, RnaFmForMaskedLM
    _USE_MULTIMOLECULE = True
except (ImportError, AttributeError):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    _USE_MULTIMOLECULE = False

# ════════════════════════════════════════════════════════════════════
# 全局配置
# ════════════════════════════════════════════════════════════════════
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("results")
MODEL_NAME = "multimolecule/rnafm"
BATCH_SIZE = 8
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# 模型加载
# ════════════════════════════════════════════════════════════════════
def load_model():
    """加载预训练 RNAFM 模型并全冻结参数（仅推理，不更新权重）。"""
    print(f"[模型] 正在从 '{MODEL_NAME}' 加载预训练权重 …")
    if _USE_MULTIMOLECULE:
        tokenizer = RnaTokenizer.from_pretrained(MODEL_NAME)
        model     = RnaFmForMaskedLM.from_pretrained(MODEL_NAME)
        print("[模型] 使用 multimolecule 专用类加载 RNA-FM")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model     = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("[模型] 使用 AutoModel 加载 RNA-FM")

    # 全冻结：不解冻、不更新任何权重
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 加载完成 | 设备: {DEVICE} | 参数量: {n_params:,} | 全部冻结")
    return tokenizer, model


# ════════════════════════════════════════════════════════════════════
# 数据处理
# ════════════════════════════════════════════════════════════════════
_N_MUTATION_PATTERN = re.compile(r"^[Nn]\d+[ACGUacgu]")   # N-notation 检测


def parse_mutations(mutant_str: str):
    """
    解析标准突变字符串，例如 'C10A,U56C' → [('C', 10, 'A'), ('U', 56, 'C')]。
    位置为 1-indexed；不处理 N-notation（返回空列表）。
    """
    result = []
    for token in re.split(r"[,;/]", mutant_str):
        token = token.strip()
        m = re.match(r"([ACGUTacgut])(\d+)([ACGUTacgut])", token)
        if m:
            wt  = m.group(1).upper().replace("T", "U")
            pos = int(m.group(2))          # 1-indexed
            mut = m.group(3).upper().replace("T", "U")
            result.append((wt, pos, mut))
    return result


def _is_n_notation_dataset(mut_df: pd.DataFrame) -> bool:
    """判断数据集是否为 N-notation 格式（如 N24A,N25G,…）。"""
    for _, row in mut_df.iterrows():
        mutant_str = str(row["mutant"]).strip()
        if mutant_str:
            first_token = re.split(r"[,;/]", mutant_str)[0].strip()
            return bool(_N_MUTATION_PATTERN.match(first_token))
    return False


def _reconstruct_wt(mut_df: pd.DataFrame) -> str | None:
    """
    当数据集没有显式野生型行时，从第一个标准突变行还原野生型序列。
    做法：取突变体序列，在所有突变位点处将突变碱基还原为野生型碱基。
    """
    for _, row in mut_df.iterrows():
        mutations = parse_mutations(str(row["mutant"]))
        if not mutations:
            continue
        seq = list(str(row["sequence"]))
        ok  = True
        for (wt_aa, pos, mut_aa) in mutations:
            idx = pos - 1
            if idx < 0 or idx >= len(seq):
                ok = False
                break
            cur = seq[idx].upper()
            if cur == mut_aa:
                seq[idx] = wt_aa       # 将突变位点还原
            else:
                ok = False             # 序列与突变注释不一致
                break
        if ok:
            return "".join(seq)
    return None


def load_dataset(csv_path: Path):
    """
    读取数据集 CSV。
    支持：
      1. 含野生型行（mutant 列为空）的数据集
      2. 无野生型行但使用标准突变注释的数据集（自动重构 WT）
    不支持（跳过）：
      - N-notation 格式数据集（野生型碱基未知）
    返回 (wt_seq, mut_df) 或 (None, None)。
    """
    df = pd.read_csv(csv_path)
    df["mutant"] = df["mutant"].fillna("").astype(str).str.strip()

    wt_mask = df["mutant"] == ""
    mut_df  = df[~wt_mask].copy().reset_index(drop=True)

    if len(mut_df) == 0:
        print(f"  [跳过] {csv_path.name}：无突变体行")
        return None, None

    # N-notation 数据集：无法确定野生型 → 跳过
    if _is_n_notation_dataset(mut_df):
        print(f"  [跳过] {csv_path.name}：N-notation 格式，野生型碱基未知")
        return None, None

    # 有显式野生型行
    if wt_mask.sum() > 0:
        wt_seq = df.loc[wt_mask, "sequence"].iloc[0]
        return wt_seq, mut_df

    # 无野生型行 → 尝试从突变重构
    wt_seq = _reconstruct_wt(mut_df)
    if wt_seq is None:
        print(f"  [跳过] {csv_path.name}：无法重构野生型序列")
        return None, None

    print(f"  [重构] 野生型序列已从突变注释还原")
    return wt_seq, mut_df


# ════════════════════════════════════════════════════════════════════
# 批量推理
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def get_logits_batch(model, tokenizer, sequences: list, batch_size: int = 8):
    """
    对一组 RNA 序列批量推理。
    返回：
      logits_list : list[Tensor(L, V)]   去掉 padding 后的真实序列长度
      ids_list    : list[Tensor(L,)]
    其中 L 包含首尾特殊 token（[CLS] 和 [EOS]）。
    """
    logits_list, ids_list = [], []

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start: start + batch_size]
        enc   = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc   = {k: v.to(DEVICE) for k, v in enc.items()}
        out   = model(**enc)          # logits: (B, L_pad, V)

        logits    = out.logits            # (B, L_pad, V)
        input_ids = enc["input_ids"]      # (B, L_pad)
        attn_mask = enc["attention_mask"] # (B, L_pad)

        for j in range(len(batch)):
            mask = attn_mask[j].bool()
            logits_list.append(logits[j][mask].cpu())   # (L_real, V)
            ids_list.append(input_ids[j][mask].cpu())   # (L_real,)

    return logits_list, ids_list


# ════════════════════════════════════════════════════════════════════
# 打分函数
# ════════════════════════════════════════════════════════════════════
def score_wt_llr(wt_logits: torch.Tensor, mutations: list, tokenizer) -> float:
    """
    WT-LLR：以野生型上下文为条件的对数似然比（无 masking 高效近似）。

    对野生型序列进行一次前向传播，提取所有位点的 logits，
    在各突变位点计算对数似然比：

        WT-LLR = Σ_{i ∈ mutations} [ log P(mut_i | WT) − log P(wt_i | WT) ]

    正值 → 模型认为突变碱基比野生型碱基更符合上下文（预测有益突变）。
    """
    log_probs = F.log_softmax(wt_logits, dim=-1)  # (L, V)
    score     = 0.0

    for (wt_aa, pos, mut_aa) in mutations:
        # 1-indexed pos → token 索引 = pos（index 0 为 [CLS]）
        tok_idx = pos
        if tok_idx >= log_probs.shape[0]:
            continue

        wt_id  = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_id = tokenizer.convert_tokens_to_ids(mut_aa)

        if wt_id  in (tokenizer.unk_token_id, None):
            continue
        if mut_id in (tokenizer.unk_token_id, None):
            continue

        score += (log_probs[tok_idx, mut_id] - log_probs[tok_idx, wt_id]).item()

    return score


def _pseudo_log_likelihood(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """
    无 masking 的伪对数似然近似：

        PLL(seq) = Σ_{i=1}^{L} log P(seq_i | seq)

    去除首尾特殊 token（[CLS] 和 [EOS]）后计算。
    """
    seq_logits = logits[1:-1]       # (L, V)
    seq_ids    = input_ids[1:-1]    # (L,)
    log_probs  = F.log_softmax(seq_logits, dim=-1)
    return log_probs[torch.arange(len(seq_ids)), seq_ids].sum().item()


def score_pll_d(wt_logits, wt_ids, mut_logits, mut_ids) -> float:
    """
    PLL-D：伪对数似然差。

    对野生型和突变体序列各进行一次前向传播：

        PLL-D = PLL(mutant) − PLL(WT)

    正值 → 突变体序列整体上更符合模型分布（预测有益突变）。
    """
    return (_pseudo_log_likelihood(mut_logits, mut_ids)
            - _pseudo_log_likelihood(wt_logits, wt_ids))


# ════════════════════════════════════════════════════════════════════
# 评价指标（不依赖 scikit-learn）
# ════════════════════════════════════════════════════════════════════
def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """梯形法计算 ROC-AUC。"""
    order  = np.argsort(scores)[::-1]
    labels = labels[order]
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tp, fp    = 0, 0
    tpr, fpr  = [0.0], [0.0]
    for lbl in labels:
        if lbl:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / pos)
        fpr.append(fp / neg)
    return float(np.trapz(tpr, fpr))


def _compute_mcc(labels: np.ndarray, preds: np.ndarray) -> float:
    """Matthews 相关系数。"""
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


def compute_metrics(scores, dms_scores) -> dict:
    """
    计算三项评价指标：
      - Spearman 相关系数（SR）
      - AUC（以中位数 DMS score 为阈值二值化）
      - MCC（以中位数预测分和中位数 DMS score 双中位数阈值）
    """
    s = np.asarray(scores,     dtype=float)
    d = np.asarray(dms_scores, dtype=float)
    valid = np.isfinite(s) & np.isfinite(d)

    if valid.sum() < 5:
        return dict(spearman=np.nan, auc=np.nan, mcc=np.nan, n=int(valid.sum()))

    s, d = s[valid], d[valid]

    sr, _     = stats.spearmanr(s, d)
    threshold = np.median(d)
    labels    = (d > threshold).astype(int)
    pred_bin  = (s > np.median(s)).astype(int)
    auc       = _compute_auc(labels, s)
    mcc       = _compute_mcc(labels, pred_bin)

    return dict(spearman=float(sr), auc=float(auc), mcc=float(mcc), n=int(valid.sum()))


# ════════════════════════════════════════════════════════════════════
# 可视化
# ════════════════════════════════════════════════════════════════════
def visualize(results_df: pd.DataFrame, out_dir: Path):
    """生成三张汇总可视化图表并保存至 out_dir。"""
    schemes = ["WT-LLR", "PLL-D"]
    metrics = ["spearman", "auc", "mcc"]
    metric_labels = {
        "spearman": "Spearman (SR)",
        "auc":      "AUC",
        "mcc":      "MCC",
    }

    # ── 图1：箱型图 —— 各指标跨数据集分布 ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "RNAFM Zero-Shot Mutation Effect Prediction\n(across all datasets)",
        fontsize=13, fontweight="bold",
    )
    for ax, metric in zip(axes, metrics):
        data = [results_df[f"{s}_{metric}"].dropna().values for s in schemes]
        ax.boxplot(
            data, labels=schemes, patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue", color="steelblue"),
            medianprops=dict(color="firebrick", linewidth=2),
            whiskerprops=dict(color="steelblue"),
            capprops=dict(color="steelblue"),
            flierprops=dict(marker="o", color="steelblue", alpha=0.5, markersize=4),
        )
        ax.set_title(metric_labels[metric], fontsize=11)
        ax.set_ylabel("Score")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "summary_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [图1] summary_boxplot.png 已保存")

    # ── 图2：热图 —— 每个数据集的详细性能 ────────────────────────────
    cols = [f"{s}_{m}" for s in schemes for m in metrics]
    hmap = results_df.set_index("dataset")[cols].astype(float)
    hmap.columns = [
        f"{s}\n{metric_labels[m]}" for s in schemes for m in metrics
    ]
    fig, ax = plt.subplots(figsize=(11, max(5, len(results_df) * 0.48)))
    sns.heatmap(
        hmap, annot=True, fmt=".3f", cmap="RdYlGn",
        center=0, linewidths=0.4, ax=ax,
        cbar_kws={"label": "Score", "shrink": 0.8},
    )
    ax.set_title("Per-Dataset Performance Heatmap", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [图2] heatmap.png 已保存")

    # ── 图3：散点图 —— WT-LLR vs PLL-D Spearman 对比 ─────────────────
    x = results_df["WT-LLR_spearman"].values.astype(float)
    y = results_df["PLL-D_spearman"].values.astype(float)
    valid = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(x[valid], y[valid], alpha=0.75, s=60, zorder=3, color="steelblue")
    for _, row in results_df.iterrows():
        xv = row["WT-LLR_spearman"]
        yv = row["PLL-D_spearman"]
        if np.isfinite(xv) and np.isfinite(yv):
            ax.annotate(
                row["dataset"][:20],
                (xv, yv), fontsize=5.5,
                ha="center", va="bottom", alpha=0.75,
            )
    finite_vals = np.concatenate([x[valid], y[valid]])
    lim = max(np.abs(finite_vals).max() + 0.1, 0.3) if len(finite_vals) > 0 else 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, alpha=0.4, label="y = x")
    ax.set_xlabel("WT-LLR  Spearman (SR)")
    ax.set_ylabel("PLL-D   Spearman (SR)")
    ax.set_title("WT-LLR vs PLL-D: Spearman Correlation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "wt_llr_vs_pll_d_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [图3] wt_llr_vs_pll_d_spearman.png 已保存")


# ════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  RNA 零样本突变效应预测  |  RNAFM 全冻结推理")
    print("  打分方案: WT-LLR  /  PLL-D")
    print("  评价指标: Spearman (SR) / AUC / MCC")
    print("=" * 65)

    tokenizer, model = load_model()

    csv_files   = sorted(DATA_DIR.glob("*.csv"))
    all_results = []

    for csv_path in tqdm(csv_files, desc="[进度]", file=sys.stdout):
        name = csv_path.stem
        print(f"\n{'─' * 55}")
        print(f"[数据集] {name}")

        wt_seq, mut_df = load_dataset(csv_path)
        if wt_seq is None:
            continue

        L = len(wt_seq)
        N = len(mut_df)
        print(f"  野生型长度: {L} nt  |  突变体数量: {N}")

        # ── 野生型一次前向传播 ──────────────────────────────────────
        print("  [推理] 野生型序列前向传播 …")
        wt_logits_list, wt_ids_list = get_logits_batch(
            model, tokenizer, [wt_seq], batch_size=1,
        )
        wt_logits = wt_logits_list[0]   # (L_wt+2, V)  含 [CLS] 和 [EOS]
        wt_ids    = wt_ids_list[0]

        # ── 所有突变体前向传播（PLL-D 需要）────────────────────────
        print(f"  [推理] {N} 条突变体序列前向传播（批大小={BATCH_SIZE}）…")
        mut_seqs = mut_df["sequence"].tolist()
        mut_logits_list, mut_ids_list = get_logits_batch(
            model, tokenizer, mut_seqs, batch_size=BATCH_SIZE,
        )

        # ── 逐突变体打分 ─────────────────────────────────────────────
        wt_llr_scores, pll_d_scores, dms_scores = [], [], []

        for i, row in mut_df.iterrows():
            mutations = parse_mutations(str(row["mutant"]))
            if mutations:
                wt_llr = score_wt_llr(wt_logits, mutations, tokenizer)
                pll_d  = score_pll_d(
                    wt_logits, wt_ids,
                    mut_logits_list[i], mut_ids_list[i],
                )
            else:
                wt_llr = np.nan
                pll_d  = np.nan

            wt_llr_scores.append(wt_llr)
            pll_d_scores.append(pll_d)
            dms_scores.append(row["DMS_score"])

        # ── 计算三项指标 ─────────────────────────────────────────────
        m_llr  = compute_metrics(wt_llr_scores, dms_scores)
        m_plld = compute_metrics(pll_d_scores,  dms_scores)

        row_result = {"dataset": name}
        for k, v in m_llr.items():
            row_result[f"WT-LLR_{k}"] = v
        for k, v in m_plld.items():
            row_result[f"PLL-D_{k}"] = v
        all_results.append(row_result)

        print(
            f"  WT-LLR → SR={m_llr['spearman']:+.3f}  "
            f"AUC={m_llr['auc']:.3f}  MCC={m_llr['mcc']:+.3f}"
            f"  (n={m_llr['n']})"
        )
        print(
            f"  PLL-D  → SR={m_plld['spearman']:+.3f}  "
            f"AUC={m_plld['auc']:.3f}  MCC={m_plld['mcc']:+.3f}"
            f"  (n={m_plld['n']})"
        )

    if not all_results:
        print("\n[错误] 没有成功处理任何数据集，请检查 data/ 目录")
        sys.exit(1)

    # ── 保存详细指标 ─────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    out_csv    = OUTPUT_DIR / "metrics.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[结果] 详细指标已保存至 {out_csv}")

    # ── 汇总统计 ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  汇总统计（所有有效数据集均值 ± 标准差）")
    print("=" * 65)
    for scheme in ["WT-LLR", "PLL-D"]:
        sr  = results_df[f"{scheme}_spearman"].dropna()
        auc = results_df[f"{scheme}_auc"].dropna()
        mcc = results_df[f"{scheme}_mcc"].dropna()
        print(
            f"  {scheme:<8}  "
            f"SR={sr.mean():+.3f}±{sr.std():.3f}  "
            f"AUC={auc.mean():.3f}±{auc.std():.3f}  "
            f"MCC={mcc.mean():+.3f}±{mcc.std():.3f}  "
            f"(n_datasets={len(sr)})"
        )

    # ── 可视化 ───────────────────────────────────────────────────────
    print("\n[可视化] 正在生成图表 …")
    visualize(results_df, OUTPUT_DIR)

    print(f"\n[完成] 所有结果已保存至 {OUTPUT_DIR}/")
    print("  - metrics.csv")
    print("  - summary_boxplot.png")
    print("  - heatmap.png")
    print("  - wt_llr_vs_pll_d_spearman.png")


if __name__ == "__main__":
    main()

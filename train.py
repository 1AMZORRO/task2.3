#!/usr/bin/env python3
"""
train.py — RNAFM Embedding 提取 + 监督回归训练 + 可视化
=========================================================

完整流程
--------
1. **加载 RNAFM 预训练模型**（冻结全部参数）
2. **提取 embedding**：对每条 RNA 序列做 mean-pooled 表示
3. **划分训练/验证集**（默认 80/20）
4. **训练线性回归头**：在固定 embedding 上用 MSE 损失预测 DMS_score
5. **记录每 epoch**：train_loss、val_loss、val_Spearman ρ
6. **可视化**：
   - 每个数据集生成独立的双子图（loss 曲线 + Spearman 曲线），保存为 PNG
   - 所有数据集汇总 Spearman 柱状图
7. **保存结果**：CSV 汇总 + 各数据集带预测值的 CSV + embedding .npy

使用方法
--------
    # 标准运行（GPU）：
    python train.py --data_dir data/ --output_dir results/

    # 快速验证（每个数据集只用前 200 条，20 个 epoch）：
    python train.py --data_dir data/ --output_dir results/ \\
        --max_seqs 200 --epochs 20

    # 自定义训练参数：
    python train.py --data_dir data/ --output_dir results/ \\
        --epochs 100 --lr 5e-4 --hidden_dim 128 --train_ratio 0.8

    # 同时计算 zero-shot PLI 基线（较慢）：
    python train.py --data_dir data/ --output_dir results/ --use_pli

参考文献
--------
- RNAFM: https://huggingface.co/multimolecule/rnafm
- MultiMolecule: https://multimolecule.danling.org
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示器服务器兼容
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from multimolecule import RnaFmForMaskedLM, RnaTokenizer
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RNAFM embedding 提取 + 监督回归训练 + 可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",   type=str, default="data",
                        help="包含 CSV 数据集的目录")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="结果保存目录")
    parser.add_argument("--model_name", type=str,
                        default="multimolecule/rnafm",
                        help="HuggingFace 模型名称或本地路径")
    parser.add_argument("--device",     type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="运行设备")
    # Embedding
    parser.add_argument("--batch_size", type=int, default=64,
                        help="embedding 提取批处理大小")
    # 训练
    parser.add_argument("--epochs",     type=int, default=50,
                        help="每个数据集的训练轮次")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Adam 优化器学习率")
    parser.add_argument("--train_ratio",type=float, default=0.8,
                        help="训练集比例（其余为验证集）")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="回归头隐藏层维度（0 = 纯线性）")
    # PLI（可选）
    parser.add_argument("--use_pli",    action="store_true",
                        help="同时计算 zero-shot 伪对数似然（PLI）基线，较慢")
    parser.add_argument("--pli_inner_batch", type=int, default=128,
                        help="PLI 计算的内部批大小")
    # 数据集选择
    parser.add_argument("--dataset",    type=str, default=None,
                        help="只训练指定数据集，传入文件名（含或不含 .csv 均可），"
                             "例如 --dataset Domingo_2018_tRNA；"
                             "不传则依次处理 data_dir 下全部数据集")
    # 数据限制
    parser.add_argument("--max_seqs",   type=int, default=None,
                        help="每个数据集最多使用的序列数（None=全部）")
    parser.add_argument("--seed",       type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 回归头网络
# ---------------------------------------------------------------------------

class RegressionHead(nn.Module):
    """
    在固定 RNAFM embedding 之上训练的小型回归网络。

    hidden_dim > 0：Linear -> ReLU -> Dropout -> Linear
    hidden_dim = 0：纯线性回归
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: torch.device):
    """加载 RNA tokenizer 和 MLM 模型（权重冻结）。"""
    print(f"  加载 tokenizer：{model_name}")
    tokenizer = RnaTokenizer.from_pretrained(model_name)

    print(f"  加载模型权重：{model_name}")
    model = RnaFmForMaskedLM.from_pretrained(model_name)
    # 冻结全部 RNAFM 参数，仅训练下游回归头
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{n_params:,}（全部冻结）")
    if device.type == "cuda":
        used  = torch.cuda.memory_allocated(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  GPU 显存：已用 {used:.2f} GB / 共 {total:.1f} GB")

    return tokenizer, model


# ---------------------------------------------------------------------------
# Embedding 提取
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    sequences: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    对 RNA 序列批量提取 mean-pooled embedding（冻结 RNAFM）。

    使用最后一层 hidden state，对所有核苷酸位置（排除 [CLS]）做均值池化。

    Returns
    -------
    embeddings : ndarray, shape (N, hidden_size)
    """
    embeddings = []
    allowed_input_keys = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    for i in tqdm(range(0, len(sequences), batch_size),
                  desc="  提取 embedding", leave=False):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=440, return_special_tokens_mask=True,
        ).to(device)
        # special_tokens_mask 仅用于池化时过滤，不是模型 forward 入参
        model_inputs = {k: v for k, v in enc.items() if k in allowed_input_keys}

        out = model.model(**model_inputs)
        hidden = out.last_hidden_state          # [B, L, H]

        # Mean pool（排除特殊 token）
        mask = enc["attention_mask"].float()    # [B, L]
        if "special_tokens_mask" in enc:
            sp_mask = enc["special_tokens_mask"].float()
            if sp_mask.shape == mask.shape:
                mask = mask * (1.0 - sp_mask)
            else:
                warnings.warn(
                    f"special_tokens_mask 与 attention_mask 形状不一致 "
                    f"({tuple(sp_mask.shape)} vs {tuple(mask.shape)})，"
                    "跳过特殊 token 过滤；这可能使特殊 token 被纳入均值池化，请检查 tokenizer 配置。"
                )
        mask_e = mask.unsqueeze(-1)             # [B, L, 1]
        emb = (hidden * mask_e).sum(1) / mask_e.sum(1).clamp(min=1e-9)  # [B, H]
        embeddings.append(emb.cpu().float().numpy())

    return np.vstack(embeddings)


# ---------------------------------------------------------------------------
# PLI 打分（可选 zero-shot 基线）
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_pli(
    sequences: list[str],
    model,
    tokenizer,
    device: torch.device,
    inner_batch: int = 128,
) -> np.ndarray:
    """
    计算每条序列的伪对数似然（PLI）。

    PLI(seq) = (1/L) * Σ_i log P(x_i | x_{-i})

    通过对每个位置逐一 mask 并用 MLM head 计算对数概率来估算。
    """
    pli_scores = []
    warned_fallback = False
    for seq in tqdm(sequences, desc="  计算 PLI（zero-shot 基线）", leave=False):
        enc = tokenizer(seq, return_tensors="pt",
                        truncation=True, max_length=440, return_special_tokens_mask=True).to(device)
        ids  = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        special = enc.get("special_tokens_mask")
        if special is not None:
            special = special[0]
            positions = torch.where((attn == 1) & (special == 0))[0].tolist()
        else:
            L = ids.size(0)
            if not warned_fallback:
                warnings.warn(
                    "tokenizer 未返回 special_tokens_mask，PLI 使用首尾 token 回退策略；"
                    "该策略未必适用于所有 tokenizer，请确认 tokenizer 配置或手动校验特殊 token 位置。"
                )
                warned_fallback = True
            positions = list(range(1, L - 1))   # 回退：排除首尾特殊 token
        if not positions:
            pli_scores.append(float("nan"))
            continue

        total = 0.0
        for i in range(0, len(positions), inner_batch):
            bp   = positions[i : i + inner_batch]
            bsz  = len(bp)
            mids = ids.unsqueeze(0).expand(bsz, -1).clone()
            for j, pos in enumerate(bp):
                mids[j, pos] = tokenizer.mask_token_id
            out      = model(input_ids=mids,
                             attention_mask=attn.unsqueeze(0).expand(bsz, -1))
            log_prob = F.log_softmax(out.logits, dim=-1)  # [bsz, L, V]
            for j, pos in enumerate(bp):
                total += log_prob[j, pos, ids[pos]].item()

        pli_scores.append(total / len(positions))

    return np.array(pli_scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def find_wildtype(df: pd.DataFrame) -> tuple[int | None, str | None]:
    """找到野生型（mutant 列为空的行）。"""
    if "mutant" not in df.columns:
        return None, None
    wt_mask = df["mutant"].isna() | (df["mutant"].astype(str).str.strip() == "")
    if wt_mask.any():
        pos = int(np.where(wt_mask.values)[0][0])
        return pos, df.iloc[pos]["sequence"]
    return None, None


def cosine_similarity_with_ref(embs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """计算所有 embedding 与参考 embedding 的余弦相似度。"""
    norms    = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
    ref_norm = max(float(np.linalg.norm(ref)), 1e-9)
    return (embs @ ref) / (norms.squeeze(axis=1) * ref_norm)


def calc_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """计算 Spearman ρ，自动过滤 NaN。"""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 5:
        return float("nan"), float("nan")
    rho, pval = spearmanr(y_true[valid], y_pred[valid])
    return float(rho), float(pval)


# ---------------------------------------------------------------------------
# 回归头训练
# ---------------------------------------------------------------------------

def train_regression(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
) -> tuple[dict, RegressionHead]:
    """
    在固定 embedding 上训练小型回归头，返回训练历史记录和训练好的模型。

    Parameters
    ----------
    embeddings : ndarray [N, H]
    labels     : ndarray [N]  — DMS_score
    train_idx  : ndarray — 训练集行索引
    val_idx    : ndarray — 验证集行索引

    Returns
    -------
    history : dict  键：train_loss, val_loss, val_spearman（均为 list，长度=epochs）
    head    : RegressionHead（已训练好）
    """
    hidden_size = embeddings.shape[1]
    head = RegressionHead(hidden_size, hidden_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr = torch.tensor(embeddings[train_idx], dtype=torch.float32).to(device)
    y_tr = torch.tensor(labels[train_idx],     dtype=torch.float32).to(device)
    X_vl = torch.tensor(embeddings[val_idx],   dtype=torch.float32).to(device)
    y_vl_np = labels[val_idx].astype(np.float32)

    history: dict[str, list[float]] = {
        "train_loss":   [],
        "val_loss":     [],
        "val_spearman": [],
    }

    for _ in tqdm(range(epochs), desc="  训练回归头", leave=False):
        # ── 训练步 ────────────────────────────────────────────────
        head.train()
        optimizer.zero_grad()
        pred_tr   = head(X_tr)
        train_loss = criterion(pred_tr, y_tr)
        train_loss.backward()
        optimizer.step()
        history["train_loss"].append(float(train_loss.item()))

        # ── 验证步 ────────────────────────────────────────────────
        head.eval()
        with torch.no_grad():
            pred_vl = head(X_vl).cpu().numpy()
        val_loss   = float(F.mse_loss(
            torch.tensor(pred_vl),
            torch.tensor(y_vl_np)).item())
        rho, _     = calc_spearman(y_vl_np, pred_vl)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(rho)

    return history, head


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    dataset_name: str,
    output_dir: Path,
    zero_shot_rho: float | None = None,
) -> None:
    """
    为单个数据集绘制双子图：
      左：Train Loss + Val Loss 曲线
      右：Val Spearman ρ 曲线（可选显示 zero-shot 基线）
    """
    epochs_x = range(1, len(history["train_loss"]) + 1)
    final_rho = history["val_spearman"][-1] if history["val_spearman"] else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(dataset_name, fontsize=13, fontweight="bold")

    # ── 左图：Loss 曲线 ──────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(epochs_x, history["train_loss"], label="Train Loss",
             color="#2196F3", linewidth=1.8)
    ax0.plot(epochs_x, history["val_loss"],   label="Val Loss",
             color="#FF5722", linewidth=1.8, linestyle="--")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("MSE Loss")
    ax0.set_title("Loss Curves")
    ax0.legend(framealpha=0.8)
    ax0.grid(True, alpha=0.3)

    # ── 右图：Spearman 曲线 ──────────────────────────────────────
    ax1 = axes[1]
    ax1.plot(epochs_x, history["val_spearman"],
             color="#4CAF50", linewidth=1.8, label=f"Val Spearman ρ")
    ax1.axhline(final_rho, color="#4CAF50", linestyle=":",
                alpha=0.6, label=f"Final: {final_rho:+.4f}")
    if zero_shot_rho is not None and not np.isnan(zero_shot_rho):
        ax1.axhline(zero_shot_rho, color="#9C27B0", linestyle="--",
                    linewidth=1.5, label=f"Zero-shot (cos): {zero_shot_rho:+.4f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Spearman ρ")
    ax1.set_title("Spearman Score")
    ax1.legend(framealpha=0.8)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(curves_dir / f"{dataset_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """绘制所有数据集最终 Spearman ρ 的横向柱状图（汇总图）。"""
    df = summary_df.copy()
    df = df.dropna(subset=["final_val_spearman"])
    if df.empty:
        return

    df = df.sort_values("final_val_spearman", ascending=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.32)))

    colors = ["#4CAF50" if v >= 0 else "#F44336"
              for v in df["final_val_spearman"]]
    bars = ax.barh(df["dataset"], df["final_val_spearman"],
                   color=colors, edgecolor="white", height=0.7)

    # 数据标签
    for bar, val in zip(bars, df["final_val_spearman"]):
        ax.text(
            val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Val Spearman ρ (final epoch)", fontsize=11)
    ax.set_title("datasets Spearman summary", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  汇总图已保存：{output_dir / 'summary_spearman.png'}")

def plot_scatter(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    dataset_name: str, 
    output_dir: Path,
    rho: float
) -> None:
    """绘制 预测值 vs 真实值 的散点图，用于排查 Spearman 负数问题"""
    plt.figure(figsize=(6, 6))
    
    # 过滤 NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t, y_p = y_true[mask], y_pred[mask]
    
    plt.scatter(y_p, y_t, alpha=0.5, color='#2196F3', edgecolors='white', s=30)
    
    # 绘制对角线参考线（理想情况）
    combined = np.concatenate([y_t, y_p])
    if len(combined) > 0:
        low, high = combined.min(), combined.max()
        plt.plot([low, high], [low, high], color='#FF5722', linestyle='--', alpha=0.7, label='Ideal (y=x)')

    plt.title(f"{dataset_name}\nSpearman ρ: {rho:+.4f}", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted DMS Score", fontsize=10)
    plt.ylabel("Actual DMS Score", fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.legend()
    
    scatter_dir = output_dir / "scatters"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(scatter_dir / f"{dataset_name}_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 单数据集处理
# ---------------------------------------------------------------------------

def process_dataset(
    csv_path: str,
    model,
    tokenizer,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    train_ratio: float,
    hidden_dim: int,
    use_pli: bool,
    pli_inner_batch: int,
    max_seqs: int | None,
    seed: int,
) -> dict:
    """
    处理单个数据集：提取 embedding → 训练回归头 → 可视化曲线 → 保存结果。

    Returns
    -------
    result : dict  包含数据集名、样本数、train/val Spearman 等指标。
    """
    name = Path(csv_path).stem
    print(f"\n{'─'*60}")
    print(f"  数据集：{name}")

    # ── 读取数据 ────────────────────────────────────────────────
    df = pd.read_csv(csv_path).dropna(subset=["sequence"]).reset_index(drop=True)
    if max_seqs and len(df) > max_seqs:
        df = df.iloc[:max_seqs].copy()
        print(f"  序列数量：{len(df)}（已截取，原始更多）")
    else:
        print(f"  序列数量：{len(df)}")

    sequences = df["sequence"].tolist()
    print(f"  序列长度：{len(sequences[0])} nt")

    result: dict = {
        "dataset":       name,
        "n_sequences":   len(df),
        "seq_length":    len(sequences[0]),
    }

    # ── Step 1：提取 embedding ───────────────────────────────────
    embeddings = extract_embeddings(sequences, model, tokenizer, device, batch_size)

    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / f"{name}.npy", embeddings.astype(np.float32))

    # ── Step 2：Zero-shot 余弦相似度基线（有 WT 时）───────────────
    wt_pos, _ = find_wildtype(df)
    has_wt    = wt_pos is not None
    result["has_wildtype"] = has_wt
    zero_shot_rho: float | None = None

    dms = df["DMS_score"].values.astype(np.float32)

    if has_wt:
        cos_scores = cosine_similarity_with_ref(embeddings, embeddings[wt_pos])
        df["cosine_score"] = cos_scores
        eval_mask = np.ones(len(df), dtype=bool)
        eval_mask[wt_pos] = False
        rho_cos, _ = calc_spearman(dms[eval_mask], cos_scores[eval_mask])
        result["zeroshot_cosine_spearman"] = rho_cos
        zero_shot_rho = rho_cos
        print(f"  [Zero-shot 余弦] Spearman ρ = {rho_cos:+.4f}  (n={eval_mask.sum()})")
    else:
        result["zeroshot_cosine_spearman"] = float("nan")

    # ── Step 3：可选 PLI zero-shot 基线 ─────────────────────────
    if use_pli:
        pli_scores = compute_pli(sequences, model, tokenizer, device, pli_inner_batch)
        df["pli_score"] = pli_scores
        pli_mask = ~np.isnan(dms)
        if has_wt:
            pli_mask[wt_pos] = False
        rho_pli, _ = calc_spearman(dms[pli_mask], pli_scores[pli_mask])
        result["zeroshot_pli_spearman"] = rho_pli
        print(f"  [Zero-shot PLI]  Spearman ρ = {rho_pli:+.4f}  (n={pli_mask.sum()})")

    # ── Step 4：划分训练/验证集 ──────────────────────────────────
    valid_idx = np.where(~np.isnan(dms))[0]
    if has_wt:
        valid_idx = valid_idx[valid_idx != wt_pos]  # 排除 WT

    n_valid = len(valid_idx)
    result["n_valid"] = n_valid

    if n_valid < 10:
        print(f"  有效样本数 {n_valid} < 10，跳过回归头训练")
        result["final_val_spearman"] = float("nan")
        return result

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(valid_idx)
    n_train   = max(2, int(n_valid * train_ratio))
    train_idx = shuffled[:n_train]
    val_idx   = shuffled[n_train:]

    if len(val_idx) < 5:
        print(f"  验证集样本数 {len(val_idx)} < 5，跳过回归头训练")
        result["final_val_spearman"] = float("nan")
        return result

    print(f"  训练集：{len(train_idx)} 条 | 验证集：{len(val_idx)} 条")

    # ── Step 5：训练回归头 ───────────────────────────────────────
    history, head = train_regression(
        embeddings, dms, train_idx, val_idx,
        epochs=epochs, lr=lr, hidden_dim=hidden_dim, device=device,
    )

    final_rho = history["val_spearman"][-1]
    best_rho  = max(history["val_spearman"])
    result["final_val_spearman"] = final_rho
    result["best_val_spearman"]  = best_rho
    result["final_train_loss"]   = history["train_loss"][-1]
    result["final_val_loss"]     = history["val_loss"][-1]
    print(f"  [回归头训练]  最终 Val Spearman ρ = {final_rho:+.4f}  "
          f"(最佳 {best_rho:+.4f}, epoch {history['val_spearman'].index(best_rho)+1})")

    # ── Step 6：绘制并保存曲线图 ─────────────────────────────────
    plot_training_curves(history, name, output_dir, zero_shot_rho=zero_shot_rho)

    # ── 保存预测 CSV ─────────────────────────────────────────────
    # 在全量 valid_idx 上做最终预测
    head.eval()
    X_all = torch.tensor(embeddings[valid_idx], dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = head(X_all).cpu().numpy()
    df.loc[df.index[valid_idx], "regression_pred"] = preds

    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_dir / f"{name}.csv", index=False)

    
    # ── Step 7：获取验证集的真实值和预测值──────────────────────────
    y_val_true = dms[val_idx]
    with torch.no_grad():
        head.eval()
        X_val = torch.tensor(embeddings[val_idx], dtype=torch.float32).to(device)
        y_val_pred = head(X_val).cpu().numpy()
    
    # 调用新写的散点图函数
    plot_scatter(y_val_true, y_val_pred, name, output_dir, final_rho)

    return result

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 设备 ───────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告：检测不到 CUDA，自动切换至 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem  = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"GPU：{gpu_name}（{gpu_mem:.1f} GB 显存）")
    else:
        print("使用 CPU 运行")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载模型 ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("加载 RNAFM 模型")
    tokenizer, model = load_model_and_tokenizer(args.model_name, device)

    # ── 发现数据集 ─────────────────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not csv_files:
        print(f"错误：在 {args.data_dir} 中未找到 CSV 文件")
        sys.exit(1)

    # 若指定了单个数据集，则只保留该文件
    if args.dataset:
        target = args.dataset if args.dataset.endswith(".csv") else args.dataset + ".csv"
        csv_files = [f for f in csv_files if os.path.basename(f) == target]
        if not csv_files:
            print(f"错误：在 {args.data_dir} 中找不到 {target}")
            print(f"可用数据集：")
            for f in sorted(glob.glob(os.path.join(args.data_dir, "*.csv"))):
                print(f"  {os.path.basename(f).replace('.csv', '')}")
            sys.exit(1)
        print(f"\n指定数据集：{os.path.basename(csv_files[0])}")
    else:
        print(f"\n找到 {len(csv_files)} 个数据集（全部处理）")

    print(f"参数：epochs={args.epochs} | lr={args.lr} | "
          f"hidden_dim={args.hidden_dim} | train_ratio={args.train_ratio}")

    # ── 逐一处理 ───────────────────────────────────────────────
    all_results: list[dict] = []
    for csv_path in csv_files:
        try:
            res = process_dataset(
                csv_path=csv_path,
                model=model,
                tokenizer=tokenizer,
                device=device,
                output_dir=output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                train_ratio=args.train_ratio,
                hidden_dim=args.hidden_dim,
                use_pli=args.use_pli,
                pli_inner_batch=args.pli_inner_batch,
                max_seqs=args.max_seqs,
                seed=args.seed,
            )
            all_results.append(res)
        except Exception as exc:
            import traceback
            print(f"\n  ✗ 处理 {csv_path} 失败：{exc}")
            traceback.print_exc()
            all_results.append({"dataset": Path(csv_path).stem, "error": str(exc)})

    # ── 汇总 ──────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_results)
    summary_path = output_dir / "results.csv"
    summary_df.to_csv(summary_path, index=False)

    # 绘制汇总 Spearman 柱状图
    plot_summary(summary_df, output_dir)

    print(f"\n{'='*60}")
    print(f"全部处理完成！共 {len(all_results)} 个数据集")
    print(f"结果目录：{output_dir}/")
    print(f"  ├── embeddings/          各数据集 embedding (.npy)")
    print(f"  ├── predictions/         带预测分数的 CSV")
    print(f"  ├── curves/              各数据集 loss + Spearman 曲线图 (.png)")
    print(f"  ├── summary_spearman.png 全数据集 Spearman 汇总图")
    print(f"  └── results.csv          汇总指标表")

    # 打印文字汇总
    show_cols = ["dataset", "n_sequences"]
    for col in ["zeroshot_cosine_spearman", "final_val_spearman", "best_val_spearman"]:
        if col in summary_df.columns:
            show_cols.append(col)
    print(f"\n{summary_df[show_cols].to_string(index=False)}")

    if "final_val_spearman" in summary_df.columns:
        valid = summary_df["final_val_spearman"].dropna()
        if len(valid):
            print(f"\n回归头 平均 Val Spearman ρ（最终 epoch）= {valid.mean():+.4f}  "
                  f"（{len(valid)}/{len(all_results)} 个数据集）")
    if "zeroshot_cosine_spearman" in summary_df.columns:
        valid = summary_df["zeroshot_cosine_spearman"].dropna()
        if len(valid):
            print(f"Zero-shot 余弦 平均 Spearman ρ          = {valid.mean():+.4f}  "
                  f"（{len(valid)}/{len(all_results)} 个含野生型数据集）")


if __name__ == "__main__":
    main()

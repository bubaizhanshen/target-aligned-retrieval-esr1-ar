#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
import yaml


@dataclass
class FoldSpec:
    fold_id: str
    train_indices: np.ndarray
    test_indices: np.ndarray


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_metric(metric_fn, y_true, y_score_or_pred) -> float:
    try:
        return float(metric_fn(y_true, y_score_or_pred))
    except Exception:
        return float("nan")


def precision_at_fraction(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    n = max(1, int(np.ceil(len(y_true) * frac)))
    order = np.argsort(y_score)[::-1][:n]
    return float(y_true[order].mean())


def enrichment_factor_at_fraction(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    base_rate = float(y_true.mean())
    if base_rate <= 0:
        return float("nan")
    return precision_at_fraction(y_true, y_score, frac) / base_rate


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "pr_auc": safe_metric(average_precision_score, y_true, y_score),
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
        "f1": safe_metric(f1_score, y_true, y_pred),
        "precision_at_5pct": precision_at_fraction(y_true, y_score, 0.05),
        "precision_at_10pct": precision_at_fraction(y_true, y_score, 0.10),
        "enrichment_factor_5pct": enrichment_factor_at_fraction(y_true, y_score, 0.05),
    }


def build_scaffold_splits(df: pd.DataFrame, tasks: list[str], cfg: dict) -> list[FoldSpec]:
    groups = df[cfg["group_column"]].fillna("MISSING").to_numpy()
    y = df["label"].astype(int).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=int(cfg["n_splits"]),
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["seed"]),
    )
    folds: list[FoldSpec] = []
    for i, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        valid = True
        for task in tasks:
            y_train = train_df[train_df["task"] == task]["label"].astype(int).to_numpy()
            y_test = test_df[test_df["task"] == task]["label"].astype(int).to_numpy()
            if len(y_train) == 0 or len(y_test) == 0:
                valid = False
                break
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                valid = False
                break
        if valid:
            folds.append(FoldSpec(f"scaffold_holdout_{i:02d}", train_idx, test_idx))
    return folds


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, descriptor: np.ndarray, fp_chunks: np.ndarray, prot_aa: np.ndarray, prot_kmer: np.ndarray, prot_meta: np.ndarray, labels: np.ndarray):
        self.descriptor = torch.tensor(descriptor, dtype=torch.float32)
        self.fp_chunks = torch.tensor(fp_chunks, dtype=torch.float32)
        self.prot_aa = torch.tensor(prot_aa, dtype=torch.float32)
        self.prot_kmer = torch.tensor(prot_kmer, dtype=torch.float32)
        self.prot_meta = torch.tensor(prot_meta, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.descriptor[idx],
            self.fp_chunks[idx],
            self.prot_aa[idx],
            self.prot_kmer[idx],
            self.prot_meta[idx],
            self.labels[idx],
        )


class CrossAttentionCPI(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, meta_dim: int):
        super().__init__()
        self.desc_proj = nn.Linear(8, hidden_dim)
        self.fp_proj = nn.Linear(32, hidden_dim)
        self.aa_proj = nn.Linear(20, hidden_dim)
        self.kmer_proj = nn.Linear(32, hidden_dim)
        self.meta_proj = nn.Linear(meta_dim, hidden_dim)
        self.comp_ln = nn.LayerNorm(hidden_dim)
        self.prot_ln = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, descriptor, fp_chunks, prot_aa, prot_kmer, prot_meta):
        desc_token = self.desc_proj(descriptor).unsqueeze(1)
        fp_tokens = self.fp_proj(fp_chunks)
        comp_tokens = self.comp_ln(torch.cat([desc_token, fp_tokens], dim=1))

        aa_token = self.aa_proj(prot_aa).unsqueeze(1)
        kmer_tokens = self.kmer_proj(prot_kmer)
        meta_token = self.meta_proj(prot_meta).unsqueeze(1)
        prot_tokens = self.prot_ln(torch.cat([aa_token, kmer_tokens, meta_token], dim=1))

        comp_ctx, _ = self.cross_attn(comp_tokens, prot_tokens, prot_tokens, need_weights=False)
        prot_ctx, _ = self.cross_attn(prot_tokens, comp_tokens, comp_tokens, need_weights=False)
        comp_ctx, _ = self.self_attn(comp_ctx, comp_ctx, comp_ctx, need_weights=False)
        prot_ctx, _ = self.self_attn(prot_ctx, prot_ctx, prot_ctx, need_weights=False)

        comp_pool = comp_ctx.mean(dim=1)
        prot_pool = prot_ctx.mean(dim=1)
        prod = comp_pool * prot_pool
        diff = torch.abs(comp_pool - prot_pool)
        return self.head(torch.cat([comp_pool, prot_pool, prod, diff], dim=1)).squeeze(-1)


def train_rf_reference(train_df: pd.DataFrame, test_df: pd.DataFrame, compound_cols: list[str], task: str, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    task_train = train_df[train_df["task"] == task].copy()
    task_test = test_df[test_df["task"] == task].copy()
    model = RandomForestClassifier(
        n_estimators=120,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(task_train[compound_cols].to_numpy(dtype=float), task_train["label"].astype(int).to_numpy())
    y_score = model.predict_proba(task_test[compound_cols].to_numpy(dtype=float))[:, 1]
    return y_score, task_test["label"].astype(int).to_numpy(), task_test["compound_id"].astype(str).tolist()


def build_arrays(df: pd.DataFrame):
    desc_cols = [c for c in df.columns if c.startswith("cmp_") and not c.startswith("cmp_fp_")]
    fp_cols = [c for c in df.columns if c.startswith("cmp_fp_")]
    aa_cols = [c for c in df.columns if c.startswith("prot_aa_")]
    kmer_cols = [c for c in df.columns if c.startswith("prot_kmer2hash_")]
    meta_cols = [c for c in df.columns if c.startswith("prot_family__")] + ["prot_seq_len_log1p"] + [c for c in df.columns if c.startswith("prot_status__")]
    # task_status onehots are absent in pair_features; keep room by padding later
    desc = df[desc_cols].to_numpy(dtype=float)
    fp = df[fp_cols].to_numpy(dtype=float)
    aa = df[aa_cols].to_numpy(dtype=float)
    kmer = df[kmer_cols].to_numpy(dtype=float)
    meta = df[meta_cols].to_numpy(dtype=float)
    return desc, fp, aa, kmer, meta, desc_cols, fp_cols, aa_cols, kmer_cols, meta_cols


def reshape_fp(fp: np.ndarray) -> np.ndarray:
    return fp.reshape(len(fp), 16, 32)


def reshape_kmer(kmer: np.ndarray) -> np.ndarray:
    return kmer.reshape(len(kmer), 4, 32)


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict, seed: int):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    desc_tr, fp_tr, aa_tr, kmer_tr, meta_tr, desc_cols, fp_cols, aa_cols, kmer_cols, meta_cols = build_arrays(train_df)
    desc_val, fp_val, aa_val, kmer_val, meta_val, *_ = build_arrays(val_df)

    desc_mean, desc_std = standardize_fit(desc_tr)
    aa_mean, aa_std = standardize_fit(aa_tr)
    meta_mean, meta_std = standardize_fit(meta_tr)

    desc_tr = standardize_apply(desc_tr, desc_mean, desc_std)
    desc_val = standardize_apply(desc_val, desc_mean, desc_std)
    aa_tr = standardize_apply(aa_tr, aa_mean, aa_std)
    aa_val = standardize_apply(aa_val, aa_mean, aa_std)
    meta_tr = standardize_apply(meta_tr, meta_mean, meta_std)
    meta_val = standardize_apply(meta_val, meta_mean, meta_std)

    y_tr = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()

    train_dataset = PairDataset(desc_tr, reshape_fp(fp_tr), aa_tr, reshape_kmer(kmer_tr), meta_tr, y_tr)
    val_dataset = PairDataset(desc_val, reshape_fp(fp_val), aa_val, reshape_kmer(kmer_val), meta_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model = CrossAttentionCPI(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        meta_dim=meta_tr.shape[1],
    ).to(device)

    pos = max(1.0, float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum())))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))

    best_state = None
    best_pr = -np.inf
    no_improve = 0

    for _epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        for desc, fp, aa, kmer, meta, y in train_loader:
            desc, fp, aa, kmer, meta, y = [x.to(device) for x in (desc, fp, aa, kmer, meta, y)]
            optimizer.zero_grad()
            logits = model(desc, fp, aa, kmer, meta)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_scores = []
        with torch.no_grad():
            for desc, fp, aa, kmer, meta, _y in val_loader:
                desc, fp, aa, kmer, meta = [x.to(device) for x in (desc, fp, aa, kmer, meta)]
                val_scores.append(torch.sigmoid(model(desc, fp, aa, kmer, meta)).cpu().numpy())
        val_scores = np.concatenate(val_scores) if val_scores else np.zeros(len(val_df), dtype=float)
        val_pr = safe_metric(average_precision_score, y_val, val_scores)
        if np.isnan(val_pr):
            val_pr = -np.inf
        if val_pr > best_pr:
            best_pr = val_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(cfg["training"]["patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu").eval()
    norms = {
        "desc_mean": desc_mean,
        "desc_std": desc_std,
        "aa_mean": aa_mean,
        "aa_std": aa_std,
        "meta_mean": meta_mean,
        "meta_std": meta_std,
    }
    return model, norms


def predict_model(model, norms, test_df: pd.DataFrame) -> np.ndarray:
    desc, fp, aa, kmer, meta, *_ = build_arrays(test_df)
    desc = standardize_apply(desc, norms["desc_mean"], norms["desc_std"])
    aa = standardize_apply(aa, norms["aa_mean"], norms["aa_std"])
    meta = standardize_apply(meta, norms["meta_mean"], norms["meta_std"])
    dataset = PairDataset(desc, reshape_fp(fp), aa, reshape_kmer(kmer), meta, test_df["label"].astype(int).to_numpy())
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    scores = []
    with torch.no_grad():
        for desc, fp, aa, kmer, meta, _y in loader:
            scores.append(torch.sigmoid(model(desc, fp, aa, kmer, meta)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(len(test_df), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-attention CPI model.")
    parser.add_argument("--config", default="configs/compound_protein_interaction_crossattn_v1.yaml")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--results-dir", default="")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    pair_dir = Path(cfg["inputs"]["pair_dataset_dir"])
    results_dir = Path(args.results_dir) if args.results_dir.strip() else Path(cfg["outputs"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pair_dir / "pair_features.csv", low_memory=False)
    tasks = list(cfg["tasks"]["core"])
    df = df[df["task"].isin(tasks)].copy()
    df["label"] = df["label"].astype(int)

    compound_cols = sorted(c for c in df.columns if c.startswith("cmp_"))
    folds = build_scaffold_splits(df, tasks, cfg["training"]["split"])
    if args.max_folds and args.max_folds > 0:
        folds = folds[: args.max_folds]
    seed = int(cfg["training"]["random_seed"])

    metric_rows = []
    pred_rows = []

    for fold_idx, fold in enumerate(folds, start=1):
        train_df = df.iloc[fold.train_indices].reset_index(drop=True)
        test_df = df.iloc[fold.test_indices].reset_index(drop=True)

        # RF reference
        for task in tasks:
            y_score, y_true, comp_ids = train_rf_reference(train_df, test_df, compound_cols, task, seed + fold_idx)
            metrics = evaluate_scores(y_true, y_score)
            metric_rows.append({"model": "rf_reference_single_task", "task": task, "split_type": "scaffold_holdout", "fold_id": fold.fold_id, **metrics})
            task_test = test_df[test_df["task"] == task].copy()
            for cid, truth, score in zip(comp_ids, y_true, y_score):
                pred_rows.append({"model": "rf_reference_single_task", "task": task, "split_type": "scaffold_holdout", "fold_id": fold.fold_id, "compound_id": cid, "y_true": int(truth), "y_score": float(score)})

        # neural model
        rng = np.random.default_rng(seed + fold_idx)
        idx = np.arange(len(train_df))
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * 0.85)))
        tr_idx, val_idx = idx[:cut], idx[cut:]
        train_sub = train_df.iloc[tr_idx].reset_index(drop=True)
        val_sub = train_df.iloc[val_idx].reset_index(drop=True)
        model, norms = train_model(train_sub, val_sub, cfg, seed + fold_idx)
        y_score_all = predict_model(model, norms, test_df)
        test_pred = test_df[["compound_id", "task", "label"]].copy()
        test_pred["y_score"] = y_score_all
        for task in tasks:
            task_df = test_pred[test_pred["task"] == task].copy()
            y_true = task_df["label"].astype(int).to_numpy()
            y_score = task_df["y_score"].astype(float).to_numpy()
            metrics = evaluate_scores(y_true, y_score)
            metric_rows.append({"model": "cross_attention_cpi", "task": task, "split_type": "scaffold_holdout", "fold_id": fold.fold_id, **metrics})
            for _, row in task_df.iterrows():
                pred_rows.append({"model": "cross_attention_cpi", "task": task, "split_type": "scaffold_holdout", "fold_id": fold.fold_id, "compound_id": row["compound_id"], "y_true": int(row["label"]), "y_score": float(row["y_score"])})

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.DataFrame(pred_rows)
    metrics_df.to_csv(results_dir / "fold_metrics.csv", index=False)
    preds_df.to_csv(results_dir / "fold_predictions.csv", index=False)
    summary = (
        metrics_df.groupby(["model", "task", "split_type"], as_index=False)[
            ["pr_auc", "roc_auc", "balanced_accuracy", "f1", "precision_at_5pct", "precision_at_10pct", "enrichment_factor_5pct"]
        ]
        .mean()
        .rename(
            columns={
                "pr_auc": "pr_auc_mean",
                "roc_auc": "roc_auc_mean",
                "balanced_accuracy": "balanced_accuracy_mean",
                "f1": "f1_mean",
                "precision_at_5pct": "precision_at_5pct_mean",
                "precision_at_10pct": "precision_at_10pct_mean",
                "enrichment_factor_5pct": "enrichment_factor_5pct_mean",
            }
        )
    )
    summary.to_csv(results_dir / "aggregate_metrics.csv", index=False)
    best = (
        summary.sort_values(["task", "pr_auc_mean"], ascending=[True, False])
        .groupby("task", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(results_dir / "best_by_task.csv", index=False)
    lines = [
        "# Cross-Attention CPI v1",
        "",
        f"- folds: `{len(folds)}`",
        f"- tasks: `{', '.join(tasks)}`",
        "",
        "## Best by task",
        "",
    ]
    for _, row in best.iterrows():
        lines.append(f"- `{row['task']}`: `{row['model']}` (PR-AUC `{row['pr_auc_mean']:.4f}`, P@5% `{row['precision_at_5pct_mean']:.4f}`, EF@5% `{row['enrichment_factor_5pct_mean']:.4f}`)")
    (results_dir / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {results_dir / 'aggregate_metrics.csv'}")
    print(f"wrote {results_dir / 'fold_predictions.csv'}")
    print(f"wrote {results_dir / 'status.md'}")


if __name__ == "__main__":
    main()

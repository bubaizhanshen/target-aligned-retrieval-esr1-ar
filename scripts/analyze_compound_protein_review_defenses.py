#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
PAIR_PRED_PATH = ROOT / "outputs/compound_protein_interaction_hybrid_paper_ready/paired_predictions_all.csv"
PAIR_FEATURES_PATH = ROOT / "data/compound_protein_interaction_v1_phase1/pair_features.csv"
V1_BANK_PATH = ROOT / "data/compound_target_retrieval_v1_bank/retrieval_bank.csv"
AR_V2_BANK_PATH = ROOT / "data/compound_target_retrieval_ar_v2_bank/retrieval_bank.csv"
OUT_DIR = ROOT / "outputs/compound_protein_interaction_review_defenses"


def safe_metric(metric_fn, y_true: np.ndarray, y_score_or_pred: np.ndarray) -> float:
    try:
        return float(metric_fn(y_true, y_score_or_pred))
    except Exception:
        return float("nan")


def precision_at_fraction(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    n = max(1, int(np.ceil(len(y_true) * frac)))
    order = np.argsort(y_score)[::-1][:n]
    return float(y_true[order].mean())


def enrichment_factor_at_fraction(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    base_rate = float(np.mean(y_true))
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
        "brier": safe_metric(brier_score_loss, y_true, y_score),
        "precision_at_5pct": precision_at_fraction(y_true, y_score, 0.05),
        "precision_at_10pct": precision_at_fraction(y_true, y_score, 0.10),
        "enrichment_factor_5pct": enrichment_factor_at_fraction(y_true, y_score, 0.05),
    }


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    ece = 0.0
    n_total = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_score = float(np.mean(y_score[mask]))
        empirical = float(np.mean(y_true[mask]))
        gap = abs(mean_score - empirical)
        ece += (n / n_total) * gap
        rows.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "n": n,
                "mean_score": mean_score,
                "empirical_positive_rate": empirical,
                "abs_gap": float(gap),
            }
        )
    return float(ece), rows


def tanimoto_similarity(test_bits: np.ndarray, train_bits_t: np.ndarray, train_bit_sums: np.ndarray) -> np.ndarray:
    intersections = test_bits @ train_bits_t
    test_sums = test_bits.sum(axis=1, keepdims=True)
    unions = test_sums + train_bit_sums[None, :] - intersections
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=np.float32), where=unions > 0)
    return sims.astype(np.float32, copy=False)


def weighted_knn_score(sims: np.ndarray, labels: np.ndarray, k: int, fallback_rate: float) -> np.ndarray:
    n_mem = sims.shape[1]
    if n_mem == 0:
        return np.full(sims.shape[0], fallback_rate, dtype=np.float32)
    k_eff = min(k, n_mem)
    top_idx = np.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    top_labels = labels[top_idx].astype(np.float32)
    weighted = (top_sims * top_labels).sum(axis=1)
    denom = top_sims.sum(axis=1)
    score = np.divide(weighted, denom, out=np.full_like(weighted, fallback_rate, dtype=np.float32), where=denom > 1e-8)
    return score.astype(np.float32, copy=False)


def top1_neighbor_info(sims: np.ndarray, train_compound_ids: np.ndarray, train_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sims.shape[1] == 0:
        n = sims.shape[0]
        return (
            np.full(n, "", dtype=object),
            np.zeros(n, dtype=np.float32),
            np.full(n, np.nan, dtype=np.float32),
        )
    best_idx = np.argmax(sims, axis=1)
    return (
        train_compound_ids[best_idx],
        sims[np.arange(len(best_idx)), best_idx].astype(np.float32),
        train_labels[best_idx].astype(np.float32),
    )


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load_pair_features() -> tuple[pd.DataFrame, list[str]]:
    probe = pd.read_csv(PAIR_FEATURES_PATH, nrows=2)
    fp_cols = [c for c in probe.columns if c.startswith("cmp_fp_")]
    usecols = [
        "compound_id",
        "task",
        "label",
        "murcko_scaffold",
        "canonical_smiles_rdkit",
        "name",
        "chemical_class",
        "dtxsid",
    ] + fp_cols
    df = pd.read_csv(PAIR_FEATURES_PATH, usecols=usecols, low_memory=False)
    df["compound_id"] = df["compound_id"].astype(str)
    df["canonical_smiles_rdkit"] = df["canonical_smiles_rdkit"].fillna("").astype(str)
    df["murcko_scaffold"] = df["murcko_scaffold"].fillna("NO_SCAFFOLD").astype(str)
    return df, fp_cols


def build_readacross_and_audits(pair_pred: pd.DataFrame, pair_df: pd.DataFrame, fp_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []

    for task in sorted(pair_pred["task"].unique()):
        task_pairs = pair_df[pair_df["task"] == task].copy()
        task_pairs = task_pairs.drop_duplicates(subset=["compound_id"]).reset_index(drop=True)
        task_bits = task_pairs[fp_cols].to_numpy(dtype=np.float32, copy=False)

        for fold_id in sorted(pair_pred.loc[pair_pred["task"] == task, "fold_id"].unique()):
            fold_pred = pair_pred[(pair_pred["task"] == task) & (pair_pred["fold_id"] == fold_id)].copy()
            test_ids = set(fold_pred["compound_id"].astype(str))
            train_mask = ~task_pairs["compound_id"].astype(str).isin(test_ids).to_numpy()
            test_mask = task_pairs["compound_id"].astype(str).isin(test_ids).to_numpy()
            train_df = task_pairs.loc[train_mask].reset_index(drop=True)
            test_df = task_pairs.loc[test_mask].reset_index(drop=True)
            train_bits = task_bits[train_mask]
            test_bits = task_bits[test_mask]

            train_sums = train_bits.sum(axis=1).astype(np.float32)
            sims = tanimoto_similarity(test_bits, train_bits.T.copy(), train_sums)
            train_labels = train_df["label"].astype(int).to_numpy()
            train_ids = train_df["compound_id"].astype(str).to_numpy()
            train_prevalence = float(np.mean(train_labels))

            knn1 = weighted_knn_score(sims, train_labels, k=1, fallback_rate=train_prevalence)
            knn5 = weighted_knn_score(sims, train_labels, k=5, fallback_rate=train_prevalence)
            knn10 = weighted_knn_score(sims, train_labels, k=10, fallback_rate=train_prevalence)
            top1_id, top1_sim, top1_label = top1_neighbor_info(sims, train_ids, train_labels)
            max_sim = sims.max(axis=1).astype(np.float32) if sims.shape[1] else np.zeros(len(test_df), dtype=np.float32)

            test_smiles = set(test_df["canonical_smiles_rdkit"].astype(str))
            train_smiles = set(train_df["canonical_smiles_rdkit"].astype(str))
            test_scaffolds = set(test_df["murcko_scaffold"].astype(str))
            train_scaffolds = set(train_df["murcko_scaffold"].astype(str))

            split_rows.append(
                {
                    "task": task,
                    "fold_id": fold_id,
                    "n_test": int(len(test_df)),
                    "n_train": int(len(train_df)),
                    "exact_smiles_overlap_count": len({s for s in test_smiles & train_smiles if s}),
                    "scaffold_overlap_count": len({s for s in test_scaffolds & train_scaffolds if s and s != "NO_SCAFFOLD"}),
                    "pct_test_maxsim_ge_0p85": float(np.mean(max_sim >= 0.85)),
                    "pct_test_maxsim_ge_0p90": float(np.mean(max_sim >= 0.90)),
                    "pct_test_maxsim_ge_0p95": float(np.mean(max_sim >= 0.95)),
                    "mean_max_train_similarity": float(np.mean(max_sim)),
                    "median_max_train_similarity": float(np.median(max_sim)),
                }
            )

            fold_pred = fold_pred.merge(
                test_df[["compound_id", "canonical_smiles_rdkit", "murcko_scaffold"]],
                on="compound_id",
                how="left",
                suffixes=("", "_pair"),
            )
            fold_pred = fold_pred.sort_values("compound_id").reset_index(drop=True)
            test_df = test_df.sort_values("compound_id").reset_index(drop=True)
            if not np.array_equal(fold_pred["compound_id"].astype(str).to_numpy(), test_df["compound_id"].astype(str).to_numpy()):
                raise RuntimeError(f"Compound alignment failed for {task} / {fold_id}")

            for i, row in fold_pred.iterrows():
                pred_rows.append(
                    {
                        "task": task,
                        "fold_id": fold_id,
                        "compound_id": row["compound_id"],
                        "y_true": int(row["y_true"]),
                        "hybrid_score": float(row["hybrid_score"]),
                        "rf_score": float(row["rf_score"]),
                        "knn1_score": float(knn1[i]),
                        "knn5_score": float(knn5[i]),
                        "knn10_score": float(knn10[i]),
                        "max_train_similarity": float(max_sim[i]),
                        "top1_neighbor_compound_id": str(top1_id[i]),
                        "top1_neighbor_similarity": float(top1_sim[i]),
                        "top1_neighbor_label": float(top1_label[i]) if not np.isnan(top1_label[i]) else np.nan,
                        "canonical_smiles_rdkit": row.get("canonical_smiles_rdkit", ""),
                        "murcko_scaffold": row.get("murcko_scaffold", "NO_SCAFFOLD"),
                    }
                )
                neighbor_rows.append(
                    {
                        "task": task,
                        "fold_id": fold_id,
                        "compound_id": row["compound_id"],
                        "y_true": int(row["y_true"]),
                        "max_train_similarity": float(max_sim[i]),
                        "top1_neighbor_compound_id": str(top1_id[i]),
                        "top1_neighbor_similarity": float(top1_sim[i]),
                        "top1_neighbor_label": float(top1_label[i]) if not np.isnan(top1_label[i]) else np.nan,
                    }
                )

    return pd.DataFrame(pred_rows), pd.DataFrame(split_rows), pd.DataFrame(neighbor_rows)


def summarize_readacross(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    for (task, fold_id), sub in pred_df.groupby(["task", "fold_id"], sort=True):
        y_true = sub["y_true"].astype(int).to_numpy()
        model_scores = {
            "hybrid": sub["hybrid_score"].to_numpy(dtype=float),
            "rf": sub["rf_score"].to_numpy(dtype=float),
            "readacross_knn1": sub["knn1_score"].to_numpy(dtype=float),
            "readacross_knn5": sub["knn5_score"].to_numpy(dtype=float),
            "readacross_knn10": sub["knn10_score"].to_numpy(dtype=float),
        }
        for model, scores in model_scores.items():
            metrics = evaluate_scores(y_true, scores)
            fold_rows.append({"task": task, "fold_id": fold_id, "model": model, **metrics})

    fold_df = pd.DataFrame(fold_rows)
    summary = (
        fold_df.groupby(["task", "model"], as_index=False)
        .agg(
            folds=("fold_id", "nunique"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            brier_mean=("brier", "mean"),
            precision_at_5pct_mean=("precision_at_5pct", "mean"),
            precision_at_10pct_mean=("precision_at_10pct", "mean"),
            enrichment_factor_5pct_mean=("enrichment_factor_5pct", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            f1_mean=("f1", "mean"),
        )
        .sort_values(["task", "pr_auc_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return fold_df, summary


def load_bank(path: Path) -> pd.DataFrame:
    base_cols = {"compound_id", "canonical_smiles_rdkit", "memory_domain", "esr1_binding", "ar_binding"}
    probe = pd.read_csv(path, nrows=1)
    usecols = [c for c in probe.columns if c in base_cols]
    bank = pd.read_csv(path, usecols=usecols, low_memory=False)
    bank["compound_id"] = bank["compound_id"].astype(str)
    bank["canonical_smiles_rdkit"] = bank["canonical_smiles_rdkit"].fillna("").astype(str)
    return bank


def build_memory_overlap_audit(pair_pred: pd.DataFrame) -> pd.DataFrame:
    bank_specs = {
        "v1_bank": load_bank(V1_BANK_PATH),
        "ar_v2_bank": load_bank(AR_V2_BANK_PATH),
    }
    rows: list[dict[str, object]] = []
    for bank_name, bank in bank_specs.items():
        for task in ["esr1_binding", "ar_binding"]:
            task_bank = bank[bank[task].notna()].copy() if task in bank.columns else bank.copy()
            bank_ids = set(task_bank["compound_id"].astype(str))
            bank_smiles = set(task_bank["canonical_smiles_rdkit"].astype(str))
            for fold_id in sorted(pair_pred.loc[pair_pred["task"] == task, "fold_id"].unique()):
                test_df = pair_pred[(pair_pred["task"] == task) & (pair_pred["fold_id"] == fold_id)]
                test_ids = set(test_df["compound_id"].astype(str))
                test_smiles = set(test_df["canonical_smiles_rdkit"].astype(str))
                alt_id_same_smiles = 0
                for _, row in test_df[["compound_id", "canonical_smiles_rdkit"]].drop_duplicates().iterrows():
                    cid = str(row["compound_id"])
                    smi = str(row["canonical_smiles_rdkit"])
                    if not smi:
                        continue
                    match = task_bank[
                        (task_bank["canonical_smiles_rdkit"].astype(str) == smi)
                        & (task_bank["compound_id"].astype(str) != cid)
                    ]
                    if not match.empty:
                        alt_id_same_smiles += 1
                rows.append(
                    {
                        "bank_name": bank_name,
                        "task": task,
                        "fold_id": fold_id,
                        "n_test": int(len(test_df)),
                        "self_compound_id_overlap_count": len(test_ids & bank_ids),
                        "exact_smiles_overlap_count_including_self": len({s for s in test_smiles & bank_smiles if s}),
                        "alt_id_same_smiles_overlap_count": int(alt_id_same_smiles),
                        "pct_test_self_compound_id_in_bank": float(len(test_ids & bank_ids) / max(1, len(test_ids))),
                        "pct_test_alt_id_same_smiles_in_bank": float(alt_id_same_smiles / max(1, len({s for s in test_ids}))),
                    }
                )
    return pd.DataFrame(rows)


def build_memory_row_flags(pair_pred: pd.DataFrame) -> pd.DataFrame:
    bank_specs = {
        "v1_bank": load_bank(V1_BANK_PATH),
        "ar_v2_bank": load_bank(AR_V2_BANK_PATH),
    }
    rows: list[dict[str, object]] = []
    for bank_name, bank in bank_specs.items():
        for task in ["esr1_binding", "ar_binding"]:
            task_bank = bank[bank[task].notna()].copy() if task in bank.columns else bank.copy()
            smiles_to_ids: dict[str, set[str]] = {}
            bank_ids = set(task_bank["compound_id"].astype(str))
            for smi, sub in task_bank.groupby("canonical_smiles_rdkit"):
                if not smi:
                    continue
                smiles_to_ids[str(smi)] = set(sub["compound_id"].astype(str))
            task_pred = pair_pred[pair_pred["task"] == task]
            for _, row in task_pred[["task", "fold_id", "compound_id", "canonical_smiles_rdkit"]].drop_duplicates().iterrows():
                cid = str(row["compound_id"])
                smi = str(row["canonical_smiles_rdkit"])
                ids = smiles_to_ids.get(smi, set())
                rows.append(
                    {
                        "bank_name": bank_name,
                        "task": task,
                        "fold_id": row["fold_id"],
                        "compound_id": cid,
                        "self_compound_in_bank": int(cid in bank_ids),
                        "alt_id_same_smiles_in_bank": int(bool(ids - {cid})),
                    }
                )
    return pd.DataFrame(rows)


def build_calibration_outputs(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    for task in sorted(pred_df["task"].unique()):
        sub = pred_df[pred_df["task"] == task]
        y_true = sub["y_true"].astype(int).to_numpy()
        for model, score_col in [("hybrid", "hybrid_score"), ("rf", "rf_score")]:
            scores = sub[score_col].to_numpy(dtype=float)
            ece, bins = expected_calibration_error(y_true, scores, n_bins=10)
            summary_rows.append(
                {
                    "task": task,
                    "model": model,
                    "brier_score": safe_metric(brier_score_loss, y_true, scores),
                    "ece_10bin": ece,
                    "mean_score": float(np.mean(scores)),
                    "positive_rate": float(np.mean(y_true)),
                }
            )
            for row in bins:
                bin_rows.append({"task": task, "model": model, **row})
    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def build_applicability_outputs(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim_bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]
    bin_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for task in sorted(pred_df["task"].unique()):
        sub = pred_df[pred_df["task"] == task].copy()
        y_true_all = sub["y_true"].astype(int).to_numpy()
        for lo, hi in sim_bins:
            mask = (sub["max_train_similarity"] >= lo) & (sub["max_train_similarity"] < hi)
            bin_df = sub[mask]
            if bin_df.empty:
                continue
            y_true = bin_df["y_true"].astype(int).to_numpy()
            bin_rows.append(
                {
                    "task": task,
                    "sim_bin": f"[{lo:.1f}, {hi:.1f})",
                    "n": int(len(bin_df)),
                    "positive_rate": float(np.mean(y_true)),
                    "hybrid_pr_auc": safe_metric(average_precision_score, y_true, bin_df["hybrid_score"].to_numpy(dtype=float)),
                    "rf_pr_auc": safe_metric(average_precision_score, y_true, bin_df["rf_score"].to_numpy(dtype=float)),
                    "hybrid_brier": safe_metric(brier_score_loss, y_true, bin_df["hybrid_score"].to_numpy(dtype=float)),
                    "rf_brier": safe_metric(brier_score_loss, y_true, bin_df["rf_score"].to_numpy(dtype=float)),
                    "mean_max_similarity": float(bin_df["max_train_similarity"].mean()),
                }
            )

        sub = sub.sort_values("max_train_similarity", ascending=False).reset_index(drop=True)
        n_total = len(sub)
        for frac in [1.0, 0.8, 0.6, 0.4, 0.2]:
            keep_n = max(1, int(np.ceil(n_total * frac)))
            keep_df = sub.iloc[:keep_n]
            y_true = keep_df["y_true"].astype(int).to_numpy()
            coverage_rows.append(
                {
                    "task": task,
                    "retained_fraction": frac,
                    "n_retained": keep_n,
                    "hybrid_pr_auc": safe_metric(average_precision_score, y_true, keep_df["hybrid_score"].to_numpy(dtype=float)),
                    "rf_pr_auc": safe_metric(average_precision_score, y_true, keep_df["rf_score"].to_numpy(dtype=float)),
                    "hybrid_brier": safe_metric(brier_score_loss, y_true, keep_df["hybrid_score"].to_numpy(dtype=float)),
                    "rf_brier": safe_metric(brier_score_loss, y_true, keep_df["rf_score"].to_numpy(dtype=float)),
                    "hybrid_precision_at_5pct": precision_at_fraction(y_true, keep_df["hybrid_score"].to_numpy(dtype=float), 0.05),
                    "rf_precision_at_5pct": precision_at_fraction(y_true, keep_df["rf_score"].to_numpy(dtype=float), 0.05),
                }
            )
    return pd.DataFrame(bin_rows), pd.DataFrame(coverage_rows)


def build_strict_no_alias_summary(pred_df: pd.DataFrame, memory_row_flags: pd.DataFrame) -> pd.DataFrame:
    task_bank_map = {
        "esr1_binding": "v1_bank",
        "ar_binding": "ar_v2_bank",
    }
    rows: list[dict[str, object]] = []
    for task, bank_name in task_bank_map.items():
        flags = memory_row_flags[
            (memory_row_flags["task"] == task)
            & (memory_row_flags["bank_name"] == bank_name)
        ][["task", "fold_id", "compound_id", "alt_id_same_smiles_in_bank"]].copy()
        merged = pred_df[pred_df["task"] == task].merge(
            flags,
            on=["task", "fold_id", "compound_id"],
            how="left",
        )
        strict = merged[merged["alt_id_same_smiles_in_bank"].fillna(0).astype(int) == 0].copy()
        if strict.empty:
            continue
        for model, score_col in [("hybrid", "hybrid_score"), ("rf", "rf_score")]:
            metrics = evaluate_scores(
                strict["y_true"].astype(int).to_numpy(),
                strict[score_col].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "task": task,
                    "bank_name": bank_name,
                    "model": model,
                    "n_rows": int(len(strict)),
                    "excluded_alias_rows": int(len(merged) - len(strict)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_status(
    readacross_summary: pd.DataFrame,
    split_audit: pd.DataFrame,
    memory_audit: pd.DataFrame,
    strict_no_alias_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    applicability_bins: pd.DataFrame,
    coverage_risk: pd.DataFrame,
) -> str:
    lines = ["# Compound-Protein Review Defense Package", ""]

    lines.append("## Read-Across Baseline")
    for task in sorted(readacross_summary["task"].unique()):
        task_df = readacross_summary[readacross_summary["task"] == task].set_index("model")
        hybrid = task_df.loc["hybrid"]
        rf = task_df.loc["rf"]
        readacross = task_df.loc["readacross_knn5"]
        lines.append(
            f"- `{task}`: hybrid `PR-AUC {hybrid['pr_auc_mean']:.4f}` vs RF `{rf['pr_auc_mean']:.4f}` vs Tanimoto kNN-5 `{readacross['pr_auc_mean']:.4f}`; hybrid `P@5% {hybrid['precision_at_5pct_mean']:.4f}` vs kNN-5 `{readacross['precision_at_5pct_mean']:.4f}`."
        )
    lines.append("")

    lines.append("## Split and Leakage Audit")
    lines.append(
        f"- Across all task/fold splits, exact train-test canonical-SMILES overlap is `{int(split_audit['exact_smiles_overlap_count'].sum())}` and nontrivial Murcko-scaffold overlap is `{int(split_audit['scaffold_overlap_count'].sum())}`."
    )
    for task in sorted(split_audit["task"].unique()):
        task_df = split_audit[split_audit["task"] == task]
        lines.append(
            f"- `{task}` nearest-neighbor difficulty: mean max train similarity `{task_df['mean_max_train_similarity'].mean():.3f}`; test compounds with max similarity `>=0.85`: `{format_pct(task_df['pct_test_maxsim_ge_0p85'].mean())}`; `>=0.90`: `{format_pct(task_df['pct_test_maxsim_ge_0p90'].mean())}`."
        )
    lines.append(
        "- Retrieval memory audit: exact test-compound IDs do appear in pollutant-aware memory banks, but the hybrid inference path explicitly excludes exact self-matches by `compound_id`; the remaining audit below quantifies exact-SMILES overlap and near-neighbor difficulty."
    )
    for bank_name in sorted(memory_audit["bank_name"].unique()):
        bank_df = memory_audit[memory_audit["bank_name"] == bank_name]
        for task in sorted(bank_df["task"].unique()):
            task_df = bank_df[bank_df["task"] == task]
            lines.append(
                f"- `{bank_name}` / `{task}`: self-ID-in-bank `{format_pct(task_df['pct_test_self_compound_id_in_bank'].mean())}` (runtime-excluded), alternate-ID same-SMILES `{format_pct(task_df['pct_test_alt_id_same_smiles_in_bank'].mean())}`."
            )
    if not strict_no_alias_summary.empty:
        lines.append(
            "- Strict alias-removed subset check: after excluding test rows whose exact canonical SMILES still appear elsewhere in the relevant memory bank, hybrid remains stronger than RF."
        )
        for task in sorted(strict_no_alias_summary["task"].unique()):
            sub = strict_no_alias_summary[strict_no_alias_summary["task"] == task].set_index("model")
            hybrid = sub.loc["hybrid"]
            rf = sub.loc["rf"]
            lines.append(
                f"- `{task}` strict no-alias subset (`n={int(hybrid['n_rows'])}`, excluded `{int(hybrid['excluded_alias_rows'])}`): hybrid `PR-AUC {hybrid['pr_auc']:.4f}` vs RF `{rf['pr_auc']:.4f}`."
            )
    lines.append("")

    lines.append("## Calibration and Applicability Domain")
    for task in sorted(calibration_summary["task"].unique()):
        task_cal = calibration_summary[calibration_summary["task"] == task].set_index("model")
        hybrid = task_cal.loc["hybrid"]
        rf = task_cal.loc["rf"]
        lines.append(
            f"- `{task}` calibration: hybrid Brier `{hybrid['brier_score']:.4f}`, ECE `{hybrid['ece_10bin']:.4f}`; RF Brier `{rf['brier_score']:.4f}`, ECE `{rf['ece_10bin']:.4f}`."
        )
        high_sim = applicability_bins[(applicability_bins["task"] == task) & (applicability_bins["sim_bin"] == "[0.8, 0.9)")]
        very_high = applicability_bins[(applicability_bins["task"] == task) & (applicability_bins["sim_bin"] == "[0.9, 1.0)")]
        if not high_sim.empty:
            row = high_sim.iloc[0]
            lines.append(
                f"- `{task}` mid-high similarity bin `[0.8,0.9)`: hybrid PR-AUC `{row['hybrid_pr_auc']:.4f}` vs RF `{row['rf_pr_auc']:.4f}` over `n={int(row['n'])}`."
            )
        if not very_high.empty:
            row = very_high.iloc[0]
            lines.append(
                f"- `{task}` very-high similarity bin `[0.9,1.0)`: hybrid PR-AUC `{row['hybrid_pr_auc']:.4f}` vs RF `{row['rf_pr_auc']:.4f}` over `n={int(row['n'])}`."
            )
        cov40 = coverage_risk[(coverage_risk["task"] == task) & (coverage_risk["retained_fraction"] == 0.4)]
        if not cov40.empty:
            row = cov40.iloc[0]
            lines.append(
                f"- `{task}` retaining the top `40%` highest-similarity compounds gives hybrid Brier `{row['hybrid_brier']:.4f}` and P@5% `{row['hybrid_precision_at_5pct']:.4f}`."
            )
    lines.append("")

    lines.append("## Files")
    lines.append("- `read_across_summary.csv`: hybrid/RF/read-across comparison.")
    lines.append("- `split_leakage_audit.csv`: train-test overlap and near-neighbor audit.")
    lines.append("- `memory_overlap_audit.csv`: overlap between evaluation compounds and retrieval memories.")
    lines.append("- `strict_no_alias_subset_summary.csv`: metrics after removing exact-SMILES alias rows from the relevant memory bank.")
    lines.append("- `calibration_summary.csv` and `calibration_bins.csv`: Brier/ECE and reliability bins.")
    lines.append("- `applicability_bin_summary.csv` and `coverage_risk_summary.csv`: similarity-stratified performance and coverage-risk summaries.")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pair_pred = pd.read_csv(PAIR_PRED_PATH, low_memory=False)
    pair_pred["compound_id"] = pair_pred["compound_id"].astype(str)
    pair_pred["canonical_smiles_rdkit"] = pair_pred["canonical_smiles_rdkit"].fillna("").astype(str)
    pair_df, fp_cols = load_pair_features()

    pred_df, split_audit, neighbor_df = build_readacross_and_audits(pair_pred, pair_df, fp_cols)
    fold_metrics, readacross_summary = summarize_readacross(pred_df)
    memory_audit = build_memory_overlap_audit(pair_pred)
    memory_row_flags = build_memory_row_flags(pair_pred)
    strict_no_alias_summary = build_strict_no_alias_summary(pred_df, memory_row_flags)
    calibration_summary, calibration_bins = build_calibration_outputs(pred_df)
    applicability_bins, coverage_risk = build_applicability_outputs(pred_df)

    pred_df.to_csv(OUT_DIR / "read_across_predictions.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / "read_across_fold_metrics.csv", index=False)
    readacross_summary.to_csv(OUT_DIR / "read_across_summary.csv", index=False)
    split_audit.to_csv(OUT_DIR / "split_leakage_audit.csv", index=False)
    neighbor_df.to_csv(OUT_DIR / "nearest_neighbor_audit_rows.csv", index=False)
    memory_audit.to_csv(OUT_DIR / "memory_overlap_audit.csv", index=False)
    memory_row_flags.to_csv(OUT_DIR / "memory_row_flags.csv", index=False)
    strict_no_alias_summary.to_csv(OUT_DIR / "strict_no_alias_subset_summary.csv", index=False)
    calibration_summary.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
    calibration_bins.to_csv(OUT_DIR / "calibration_bins.csv", index=False)
    applicability_bins.to_csv(OUT_DIR / "applicability_bin_summary.csv", index=False)
    coverage_risk.to_csv(OUT_DIR / "coverage_risk_summary.csv", index=False)

    status = build_status(
        readacross_summary=readacross_summary,
        split_audit=split_audit,
        memory_audit=memory_audit,
        strict_no_alias_summary=strict_no_alias_summary,
        calibration_summary=calibration_summary,
        applicability_bins=applicability_bins,
        coverage_risk=coverage_risk,
    )
    (OUT_DIR / "status.md").write_text(status, encoding="utf-8")


if __name__ == "__main__":
    main()

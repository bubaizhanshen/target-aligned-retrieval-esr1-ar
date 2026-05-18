#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/public/home/zhqi_hjxy/HCML")
OUT_DIR = ROOT / "outputs" / "compound_protein_interaction_hybrid_paper_ready"


TASK_SPECS = {
    "esr1_binding": {
        "pred_path": ROOT / "outputs" / "compound_protein_interaction_retrievalhybrid_v1_full5" / "fold_predictions.csv",
        "agg_path": ROOT / "outputs" / "compound_protein_interaction_retrievalhybrid_v1_full5" / "aggregate_metrics.csv",
        "bank_path": ROOT / "data" / "compound_target_retrieval_v1_bank" / "retrieval_bank.csv",
        "memories": ["pollutant_only", "generic_bioactive", "mixed"],
        "top_k": 15,
    },
    "ar_binding": {
        "pred_path": ROOT / "outputs" / "compound_protein_interaction_retrievalhybrid_ar_v2_support" / "fold_predictions.csv",
        "agg_path": ROOT / "outputs" / "compound_protein_interaction_retrievalhybrid_ar_v2_support" / "aggregate_metrics.csv",
        "bank_path": ROOT / "data" / "compound_target_retrieval_ar_v2_bank" / "retrieval_bank.csv",
        "memories": ["pollutant_only", "natural_product", "ar_support_v2"],
        "top_k": 15,
    },
}


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order].astype(float)
    pos = y.sum()
    if pos <= 0:
        return float("nan")
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / pos)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_rank_sum = ranks[y_true == 1].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def precision_at_fraction(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    n = max(1, int(math.ceil(len(y_true) * frac)))
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].mean())


def enrichment_factor_at_fraction(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    base = float(y_true.mean())
    if base <= 0:
        return float("nan")
    return precision_at_fraction(y_true, y_score, frac) / base


def balanced_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_score >= threshold).astype(int)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return float("nan")
    tpr = float((y_pred[pos_mask] == 1).mean())
    tnr = float((y_pred[neg_mask] == 0).mean())
    return (tpr + tnr) / 2


def f1_score(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": safe_average_precision(y_true, y_score),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "balanced_accuracy": balanced_accuracy(y_true, y_score),
        "f1": f1_score(y_true, y_score),
        "p_at_5": precision_at_fraction(y_true, y_score, 0.05),
        "p_at_10": precision_at_fraction(y_true, y_score, 0.10),
        "ef_at_5": enrichment_factor_at_fraction(y_true, y_score, 0.05),
    }


def classify_chemotype(name: str, smiles: str) -> str:
    n = (name or "").lower()
    s = (smiles or "").lower()
    if "paraben" in n:
        return "paraben"
    if "phthalate" in n or "phthalic" in n:
        return "phthalate"
    if "bisphenol" in n:
        return "bisphenol"
    if "benzophenone" in n:
        return "benzophenone"
    if any(tok in n for tok in ["perfluoro", "fluorotelomer", "pfos", "pfoa", "genx"]):
        return "pfas_like"
    if any(tok in n for tok in ["estradiol", "estrone", "testosterone", "progesterone", "androst", "steroid"]):
        return "steroid_like"
    if "musk" in n:
        return "musk"
    if "chlordane" in n:
        return "organochlorine"
    if "phenol red" in n:
        return "phenol_red"
    if "phenol" in n:
        return "phenol_like"
    if s.count("f") >= 8:
        return "fluorinated"
    return "other"


def load_meta() -> pd.DataFrame:
    meta = pd.read_csv(
        ROOT / "data" / "compound_protein_interaction_v1_phase1" / "pair_features.csv",
        usecols=["compound_id", "task", "name", "chemical_class", "murcko_scaffold", "canonical_smiles_rdkit"],
        low_memory=False,
    ).drop_duplicates(subset=["compound_id", "task"])
    meta["chemotype"] = [
        classify_chemotype(n, s)
        for n, s in zip(meta["name"].fillna(""), meta["canonical_smiles_rdkit"].fillna(""), strict=False)
    ]
    return meta


def load_task_predictions(task: str, meta: pd.DataFrame) -> pd.DataFrame:
    pred_path = TASK_SPECS[task]["pred_path"]
    df = pd.read_csv(pred_path)
    rf = (
        df[(df["task"] == task) & (df["model"] == "rf_reference_single_task")][["fold_id", "compound_id", "y_true", "y_score"]]
        .rename(columns={"y_score": "rf_score"})
    )
    hybrid = (
        df[(df["task"] == task) & (df["model"] == "retrieval_augmented_seq_cpi")][["fold_id", "compound_id", "y_true", "y_score"]]
        .rename(columns={"y_score": "hybrid_score"})
    )
    merged = hybrid.merge(rf, on=["fold_id", "compound_id", "y_true"], how="inner")
    merged["task"] = task
    merged["delta_score"] = merged["hybrid_score"] - merged["rf_score"]
    merged["rf_pred"] = (merged["rf_score"] >= 0.5).astype(int)
    merged["hybrid_pred"] = (merged["hybrid_score"] >= 0.5).astype(int)
    merged = merged.merge(meta[meta["task"] == task], on=["compound_id", "task"], how="left")
    return merged


def bootstrap_deltas(task_df: pd.DataFrame, n_boot: int = 2000, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = task_df["y_true"].to_numpy(dtype=int)
    rf = task_df["rf_score"].to_numpy(dtype=float)
    hyb = task_df["hybrid_score"].to_numpy(dtype=float)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(task_df), len(task_df))
        y_b = y[idx]
        rf_b = rf[idx]
        hyb_b = hyb[idx]
        rf_metrics = evaluate_binary(y_b, rf_b)
        hyb_metrics = evaluate_binary(y_b, hyb_b)
        rows.append(
            {
                key: hyb_metrics[key] - rf_metrics[key]
                for key in ["pr_auc", "roc_auc", "balanced_accuracy", "f1", "p_at_5", "p_at_10", "ef_at_5"]
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap(task: str, deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in deltas.columns:
        vals = deltas[metric].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "task": task,
                "metric": metric,
                "delta_mean": float(np.mean(vals)),
                "ci_low": float(np.quantile(vals, 0.025)),
                "ci_high": float(np.quantile(vals, 0.975)),
                "prob_gt_zero": float((vals > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def fold_level_summary(task_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_id, fold_df in task_df.groupby("fold_id"):
        rf_metrics = evaluate_binary(fold_df["y_true"].to_numpy(dtype=int), fold_df["rf_score"].to_numpy(dtype=float))
        hyb_metrics = evaluate_binary(fold_df["y_true"].to_numpy(dtype=int), fold_df["hybrid_score"].to_numpy(dtype=float))
        row = {"task": fold_df["task"].iloc[0], "fold_id": fold_id}
        for metric in rf_metrics:
            row[f"rf_{metric}"] = rf_metrics[metric]
            row[f"hybrid_{metric}"] = hyb_metrics[metric]
            row[f"delta_{metric}"] = hyb_metrics[metric] - rf_metrics[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def chemotype_summary(task_df: pd.DataFrame, min_n: int = 20, min_pos: int = 5) -> pd.DataFrame:
    rows = []
    for chemotype, sub in task_df.groupby("chemotype"):
        y = sub["y_true"].to_numpy(dtype=int)
        n = len(sub)
        pos = int(y.sum())
        neg = int(n - pos)
        if n < min_n or pos < min_pos or neg < min_pos:
            continue
        rf_metrics = evaluate_binary(y, sub["rf_score"].to_numpy(dtype=float))
        hyb_metrics = evaluate_binary(y, sub["hybrid_score"].to_numpy(dtype=float))
        rows.append(
            {
                "task": sub["task"].iloc[0],
                "chemotype": chemotype,
                "n": n,
                "positives": pos,
                "rf_pr_auc": rf_metrics["pr_auc"],
                "hybrid_pr_auc": hyb_metrics["pr_auc"],
                "delta_pr_auc": hyb_metrics["pr_auc"] - rf_metrics["pr_auc"],
                "rf_p5": rf_metrics["p_at_5"],
                "hybrid_p5": hyb_metrics["p_at_5"],
                "delta_p5": hyb_metrics["p_at_5"] - rf_metrics["p_at_5"],
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "delta_pr_auc"], ascending=[True, False])


def memory_mask(bank: pd.DataFrame, memory_name: str) -> np.ndarray:
    special = {
        "pollutant_only": "memory_eligible_pollutant_only",
        "generic_bioactive": "memory_eligible_generic_bioactive",
        "mixed": "memory_eligible_mixed",
        "natural_product": "memory_eligible_natural_product",
        "ar_support_v2": "memory_eligible_ar_support_v2",
    }
    col = special.get(memory_name, f"memory_eligible_{memory_name}")
    if col not in bank.columns:
        raise KeyError(f"Missing memory column: {col}")
    return bank[col].fillna(0).astype(int).to_numpy(dtype=bool)


def tanimoto_top_refs(
    query_fp: np.ndarray,
    bank: pd.DataFrame,
    fp_cols: list[str],
    task: str,
    memory_name: str,
    top_k: int,
    exclude_compound_id: str,
) -> pd.DataFrame:
    mask = memory_mask(bank, memory_name) & bank[task].notna().to_numpy()
    sub = bank.loc[mask].copy()
    sub = sub[sub["compound_id"].astype(str) != str(exclude_compound_id)].copy()
    if sub.empty:
        return sub
    bank_fp = sub[fp_cols].to_numpy(dtype=np.float32)
    inter = bank_fp @ query_fp
    union = bank_fp.sum(axis=1) + query_fp.sum() - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)
    sub["tanimoto"] = sim
    sub = sub.sort_values("tanimoto", ascending=False).head(top_k).copy()
    sub["memory_name"] = memory_name
    return sub[
        ["memory_name", "compound_id", "name", "domain", "reference_group", "label_source", task, "tanimoto", "murcko_scaffold"]
    ]


def retrieval_support_examples(task: str, task_df: pd.DataFrame, n_cases: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = TASK_SPECS[task]
    bank = pd.read_csv(spec["bank_path"], low_memory=False)
    fp_cols_bank = [c for c in bank.columns if c.startswith("fp_")]
    pair = pd.read_csv(ROOT / "data" / "compound_protein_interaction_v1_phase1" / "pair_features.csv", low_memory=False)
    pair = pair[pair["task"] == task].drop_duplicates(subset=["compound_id"])
    fp_cols_pair = [c for c in pair.columns if c.startswith("cmp_fp_")]
    pair_fp = pair.set_index("compound_id")[fp_cols_pair].astype(np.float32)

    rescued = task_df[(task_df["y_true"] == 1) & (task_df["rf_pred"] == 0) & (task_df["hybrid_pred"] == 1)].copy()
    rescued = rescued.sort_values("delta_score", ascending=False).head(n_cases)
    case_rows = []
    summary_rows = []
    for _, row in rescued.iterrows():
        cid = str(row["compound_id"])
        q_fp = pair_fp.loc[cid].to_numpy(dtype=np.float32)
        for memory_name in spec["memories"]:
            refs = tanimoto_top_refs(q_fp, bank, fp_cols_bank, task, memory_name, 5, cid)
            if refs.empty:
                continue
            refs = refs.copy()
            refs["task"] = task
            refs["query_compound_id"] = cid
            refs["query_name"] = row["name"]
            refs["query_delta_score"] = row["delta_score"]
            case_rows.append(refs)
            top = refs.iloc[0]
            pos_mean = float(refs[task].astype(float).mean())
            summary_rows.append(
                {
                    "task": task,
                    "query_compound_id": cid,
                    "query_name": row["name"],
                    "memory_name": memory_name,
                    "top_ref_name": top["name"],
                    "top_ref_compound_id": top["compound_id"],
                    "top_ref_similarity": float(top["tanimoto"]),
                    "mean_ref_label": pos_mean,
                    "query_delta_score": float(row["delta_score"]),
                }
            )
    return pd.concat(case_rows, ignore_index=True), pd.DataFrame(summary_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta()

    all_task_rows = []
    bootstrap_rows = []
    fold_rows = []
    chemotype_rows = []
    support_rows = []
    support_summary_rows = []

    for task in ["esr1_binding", "ar_binding"]:
        task_df = load_task_predictions(task, meta)
        task_df.to_csv(OUT_DIR / f"{task}_paired_predictions.csv", index=False)
        all_task_rows.append(task_df)

        boot = bootstrap_deltas(task_df)
        boot.to_csv(OUT_DIR / f"{task}_bootstrap_deltas.csv", index=False)
        bootstrap_rows.append(summarize_bootstrap(task, boot))

        fold_df = fold_level_summary(task_df)
        fold_df.to_csv(OUT_DIR / f"{task}_fold_level_summary.csv", index=False)
        fold_rows.append(fold_df)

        chem_df = chemotype_summary(task_df)
        chem_df.to_csv(OUT_DIR / f"{task}_chemotype_summary.csv", index=False)
        chemotype_rows.append(chem_df)

        support_df, support_summary = retrieval_support_examples(task, task_df)
        support_df.to_csv(OUT_DIR / f"{task}_retrieval_support_examples.csv", index=False)
        support_summary.to_csv(OUT_DIR / f"{task}_retrieval_support_summary.csv", index=False)
        support_rows.append(support_df)
        support_summary_rows.append(support_summary)

    paired_df = pd.concat(all_task_rows, ignore_index=True)
    paired_df.to_csv(OUT_DIR / "paired_predictions_all.csv", index=False)
    bootstrap_summary = pd.concat(bootstrap_rows, ignore_index=True)
    bootstrap_summary.to_csv(OUT_DIR / "bootstrap_summary.csv", index=False)
    fold_summary = pd.concat(fold_rows, ignore_index=True)
    fold_summary.to_csv(OUT_DIR / "fold_level_summary.csv", index=False)
    chemotype_summary_df = pd.concat(chemotype_rows, ignore_index=True)
    chemotype_summary_df.to_csv(OUT_DIR / "chemotype_summary_all.csv", index=False)
    pd.concat(support_rows, ignore_index=True).to_csv(OUT_DIR / "retrieval_support_examples_all.csv", index=False)
    pd.concat(support_summary_rows, ignore_index=True).to_csv(OUT_DIR / "retrieval_support_summary_all.csv", index=False)

    lines = ["# Hybrid Paper-Ready Analysis", ""]
    for task in ["esr1_binding", "ar_binding"]:
        task_pairs = paired_df[paired_df["task"] == task]
        rf = evaluate_binary(task_pairs["y_true"].to_numpy(dtype=int), task_pairs["rf_score"].to_numpy(dtype=float))
        hyb = evaluate_binary(task_pairs["y_true"].to_numpy(dtype=int), task_pairs["hybrid_score"].to_numpy(dtype=float))
        boot = bootstrap_summary[bootstrap_summary["task"] == task].set_index("metric")
        lines.append(f"## {task}")
        lines.append(f"- RF PR-AUC: `{rf['pr_auc']:.4f}`")
        lines.append(f"- Hybrid PR-AUC: `{hyb['pr_auc']:.4f}`")
        lines.append(
            f"- Delta PR-AUC bootstrap 95% CI: `{boot.loc['pr_auc', 'ci_low']:.4f}` to `{boot.loc['pr_auc', 'ci_high']:.4f}`"
        )
        lines.append(
            f"- Delta P@5 bootstrap 95% CI: `{boot.loc['p_at_5', 'ci_low']:.4f}` to `{boot.loc['p_at_5', 'ci_high']:.4f}`"
        )
        rescued = int(((task_pairs["y_true"] == 1) & (task_pairs["rf_pred"] == 0) & (task_pairs["hybrid_pred"] == 1)).sum())
        regressed = int(((task_pairs["y_true"] == 1) & (task_pairs["rf_pred"] == 1) & (task_pairs["hybrid_pred"] == 0)).sum())
        lines.append(f"- rescued positives: `{rescued}`")
        lines.append(f"- regressed positives: `{regressed}`")
        chem = chemotype_summary_df[chemotype_summary_df["task"] == task].head(5)
        if not chem.empty:
            lines.append("- top chemotypes by delta PR-AUC:")
            for _, row in chem.iterrows():
                lines.append(
                    f"  - `{row['chemotype']}`: delta PR-AUC `{row['delta_pr_auc']:.4f}` (`n={int(row['n'])}`, `pos={int(row['positives'])}`)"
                )
        lines.append("")
    (OUT_DIR / "status.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

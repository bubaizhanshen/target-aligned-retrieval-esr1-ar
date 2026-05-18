#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier


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


def build_scaffold_splits(df: pd.DataFrame, task_list: list[str], cfg: dict) -> list[FoldSpec]:
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
        for task in task_list:
            train_task = train_df[train_df["task"] == task]["label"].astype(int).to_numpy()
            test_task = test_df[test_df["task"] == task]["label"].astype(int).to_numpy()
            if len(train_task) == 0 or len(test_task) == 0:
                valid = False
                break
            if len(np.unique(train_task)) < 2 or len(np.unique(test_task)) < 2:
                valid = False
                break
        if valid:
            folds.append(FoldSpec(f"scaffold_holdout_{i:02d}", train_idx, test_idx))
    return folds


def standardize_fit(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def balanced_resample(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y
    if len(pos_idx) == len(neg_idx):
        return X, y
    rng = np.random.default_rng(seed)
    if len(pos_idx) < len(neg_idx):
        extra = rng.choice(pos_idx, size=len(neg_idx) - len(pos_idx), replace=True)
    else:
        extra = rng.choice(neg_idx, size=len(pos_idx) - len(neg_idx), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    rng.shuffle(idx)
    return X[idx], y[idx]


class FeatureBuilder:
    def __init__(self, compound_cols: list[str], protein_cols: list[str]):
        self.compound_cols = compound_cols
        self.protein_cols = protein_cols
        self.descriptor_cols = [c for c in compound_cols if not c.startswith("cmp_fp_")]
        self.fp_cols = [c for c in compound_cols if c.startswith("cmp_fp_")]
        self.protein_rp: np.ndarray | None = None
        self.compound_svd: TruncatedSVD | None = None
        self.compound_mean: np.ndarray | None = None
        self.compound_std: np.ndarray | None = None
        self.descriptor_mean: np.ndarray | None = None
        self.descriptor_std: np.ndarray | None = None
        self.protein_mean: np.ndarray | None = None
        self.protein_std: np.ndarray | None = None
        self.protein_proj_dim = 32
        self.compound_proj_dim = 32

    def fit(self, train_df: pd.DataFrame, seed: int) -> None:
        compound_x = train_df[self.compound_cols].to_numpy(dtype=float)
        protein_x = train_df[self.protein_cols].to_numpy(dtype=float)
        descriptor_x = train_df[self.descriptor_cols].to_numpy(dtype=float)
        self.compound_mean, self.compound_std = standardize_fit(compound_x)
        self.descriptor_mean, self.descriptor_std = standardize_fit(descriptor_x)
        self.protein_mean, self.protein_std = standardize_fit(protein_x)

        fp_train = train_df[self.fp_cols].to_numpy(dtype=float)
        n_comp = min(self.compound_proj_dim, fp_train.shape[1] - 1, max(2, fp_train.shape[0] - 1))
        self.compound_svd = TruncatedSVD(n_components=max(2, n_comp), random_state=seed)
        self.compound_svd.fit(fp_train)

        rng = np.random.default_rng(seed)
        self.protein_rp = rng.normal(0.0, 1.0, size=(len(self.protein_cols), self.protein_proj_dim))
        norms = np.linalg.norm(self.protein_rp, axis=0, keepdims=True)
        norms[norms < 1e-8] = 1.0
        self.protein_rp = self.protein_rp / norms

    def transform_blocks(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if (
            self.compound_mean is None
            or self.descriptor_mean is None
            or self.protein_mean is None
            or self.compound_svd is None
            or self.protein_rp is None
        ):
            raise RuntimeError("FeatureBuilder must be fitted before transform.")
        compound_x = standardize_apply(df[self.compound_cols].to_numpy(dtype=float), self.compound_mean, self.compound_std)
        protein_x = standardize_apply(df[self.protein_cols].to_numpy(dtype=float), self.protein_mean, self.protein_std)
        descriptor_x = standardize_apply(
            df[self.descriptor_cols].to_numpy(dtype=float),
            self.descriptor_mean,
            self.descriptor_std,
        )
        fp_x = df[self.fp_cols].to_numpy(dtype=float)
        compound_proj = self.compound_svd.transform(fp_x)
        compound_latent = np.hstack([descriptor_x, compound_proj])
        protein_latent = protein_x @ self.protein_rp
        return {
            "compound": compound_x,
            "protein": protein_x,
            "compound_descriptor": descriptor_x.astype(float),
            "compound_proj": compound_proj.astype(float),
            "compound_latent": compound_latent.astype(float),
            "protein_latent": protein_latent.astype(float),
        }


def build_model_features(model_name: str, blocks: dict[str, np.ndarray]) -> np.ndarray:
    compound_x = blocks["compound"]
    protein_x = blocks["protein"]
    compound_descriptor = blocks["compound_descriptor"]
    compound_proj = blocks["compound_proj"]
    compound_latent = blocks["compound_latent"]
    protein_latent = blocks["protein_latent"]
    if model_name == "compound_only":
        return compound_x
    if model_name == "concat_mlp":
        return np.hstack([compound_x, protein_x])
    if model_name == "bilinear":
        return np.hstack([compound_proj, protein_latent, compound_proj * protein_latent, compound_descriptor])
    if model_name == "target_aware_gating":
        gate = sigmoid(protein_latent)
        gated = compound_proj * gate
        return np.hstack([gated, protein_latent, compound_descriptor])
    if model_name == "cross_interaction_mlp":
        prod = compound_proj * protein_latent
        diff = np.abs(compound_proj - protein_latent)
        return np.hstack([compound_proj, protein_latent, prod, diff, compound_descriptor])
    raise ValueError(f"Unsupported model: {model_name}")


def train_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray, seed: int):
    X_bal, y_bal = balanced_resample(X_train, y_train, seed=seed)
    if model_name in {"bilinear", "target_aware_gating", "cross_interaction_mlp"}:
        model = LogisticRegression(
            solver="liblinear",
            max_iter=400,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    model.fit(X_bal, y_bal)
    return model


def evaluate_task(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
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


def run_rf_reference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: str,
    compound_cols: list[str],
    seed: int,
) -> pd.DataFrame:
    task_train = train_df[train_df["task"] == task].copy()
    task_test = test_df[test_df["task"] == task].copy()
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(task_train[compound_cols].to_numpy(dtype=float), task_train["label"].astype(int).to_numpy())
    score = model.predict_proba(task_test[compound_cols].to_numpy(dtype=float))[:, 1]
    out = task_test[["pair_id", "compound_id", "gene_symbol", "task", "label"]].copy()
    out["model"] = "rf_reference_single_task"
    out["split_type"] = "scaffold_holdout"
    out["y_true"] = out["label"].astype(int)
    out["y_score"] = score.astype(float)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 compound-protein interaction baselines.")
    parser.add_argument(
        "--config",
        default="configs/compound_protein_interaction_v1.yaml",
        help="Path to the compound-protein interaction YAML config.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated subset of pair models to run, excluding rf_reference_single_task.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Optional cap on the number of scaffold folds to run.",
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help="Optional override for output results directory.",
    )
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    dataset_dir = Path(cfg["inputs"]["pair_dataset_dir"])
    results_dir = Path(args.results_dir) if args.results_dir.strip() else Path(cfg["outputs"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_dir / "pair_features.csv", low_memory=False)
    core_tasks = list(cfg["tasks"]["core"])
    df = df[df["task"].isin(core_tasks)].copy()
    df["label"] = df["label"].astype(int)

    compound_cols = sorted(c for c in df.columns if c.startswith("cmp_"))
    protein_cols = sorted(c for c in df.columns if c.startswith("prot_"))
    model_names = [m for m in cfg["interaction_model"]["baseline_modes"] + cfg["interaction_model"]["proposed_modes"] if m != "rf_reference_single_task"]
    if args.models.strip():
        keep = {token.strip() for token in args.models.split(",") if token.strip()}
        model_names = [m for m in model_names if m in keep]
    seed = int(cfg["training"].get("random_seed", 17))
    folds = build_scaffold_splits(df, core_tasks, cfg["training"]["splits"]["scaffold_holdout"])
    if not folds:
        raise RuntimeError("No valid scaffold folds were generated.")
    if args.max_folds and args.max_folds > 0:
        folds = folds[: args.max_folds]

    metric_rows: list[dict] = []
    pred_rows: list[dict] = []

    for fold_idx, fold in enumerate(folds, start=1):
        train_df = df.iloc[fold.train_indices].reset_index(drop=True)
        test_df = df.iloc[fold.test_indices].reset_index(drop=True)

        for task in core_tasks:
            rf_pred = run_rf_reference(
                train_df=train_df,
                test_df=test_df,
                task=task,
                compound_cols=compound_cols,
                seed=seed + fold_idx,
            )
            y_true = rf_pred["y_true"].to_numpy(dtype=int)
            y_score = rf_pred["y_score"].to_numpy(dtype=float)
            metrics = evaluate_task(y_true, y_score)
            metric_rows.append(
                {
                    "model": "rf_reference_single_task",
                    "task": task,
                    "split_type": "scaffold_holdout",
                    "fold_id": fold.fold_id,
                    **metrics,
                }
            )
            rf_pred["fold_id"] = fold.fold_id
            pred_rows.extend(rf_pred.to_dict("records"))

        builder = FeatureBuilder(compound_cols=compound_cols, protein_cols=protein_cols)
        builder.fit(train_df, seed=seed + fold_idx)
        train_blocks = builder.transform_blocks(train_df)
        test_blocks = builder.transform_blocks(test_df)
        y_train = train_df["label"].astype(int).to_numpy()

        for model_name in model_names:
            X_train = build_model_features(model_name, train_blocks)
            model = train_model(model_name, X_train, y_train, seed=seed + fold_idx)
            X_test = build_model_features(model_name, test_blocks)
            score = model.predict_proba(X_test)[:, 1]

            test_with_scores = test_df[["pair_id", "compound_id", "gene_symbol", "task", "label"]].copy()
            test_with_scores["y_true"] = test_with_scores["label"].astype(int)
            test_with_scores["y_score"] = score.astype(float)
            test_with_scores["model"] = model_name
            test_with_scores["split_type"] = "scaffold_holdout"
            test_with_scores["fold_id"] = fold.fold_id

            for task in core_tasks:
                task_df = test_with_scores[test_with_scores["task"] == task].copy()
                y_true = task_df["y_true"].to_numpy(dtype=int)
                y_score = task_df["y_score"].to_numpy(dtype=float)
                metrics = evaluate_task(y_true, y_score)
                metric_rows.append(
                    {
                        "model": model_name,
                        "task": task,
                        "split_type": "scaffold_holdout",
                        "fold_id": fold.fold_id,
                        **metrics,
                    }
                )
            pred_rows.extend(test_with_scores.to_dict("records"))

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.DataFrame(pred_rows)
    metrics_df.to_csv(results_dir / "fold_metrics.csv", index=False)
    preds_df.to_csv(results_dir / "fold_predictions.csv", index=False)

    summary = (
        metrics_df.groupby(["model", "task", "split_type"], as_index=False)[
            [
                "pr_auc",
                "roc_auc",
                "balanced_accuracy",
                "f1",
                "precision_at_5pct",
                "precision_at_10pct",
                "enrichment_factor_5pct",
            ]
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

    best_by_task = (
        summary.sort_values(["task", "pr_auc_mean"], ascending=[True, False])
        .groupby("task", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_task.to_csv(results_dir / "best_by_task.csv", index=False)

    status = {
        "study_id": cfg["study"]["id"],
        "n_pairs": int(len(df)),
        "n_folds": int(len(folds)),
        "tasks": core_tasks,
        "models": ["rf_reference_single_task", *model_names],
    }
    (results_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Compound-Protein Interaction Phase 1 Baselines",
        "",
        f"- pairs: `{len(df)}`",
        f"- folds: `{len(folds)}`",
        f"- tasks: `{', '.join(core_tasks)}`",
        "",
        "## Best by task",
        "",
    ]
    for _, row in best_by_task.iterrows():
        lines.append(
            f"- `{row['task']}`: `{row['model']}` "
            f"(PR-AUC `{row['pr_auc_mean']:.4f}`, "
            f"P@5% `{row['precision_at_5pct_mean']:.4f}`, "
            f"EF@5% `{row['enrichment_factor_5pct_mean']:.4f}`)"
        )
    (results_dir / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {results_dir / 'aggregate_metrics.csv'}")
    print(f"wrote {results_dir / 'fold_predictions.csv'}")
    print(f"wrote {results_dir / 'status.md'}")


if __name__ == "__main__":
    main()

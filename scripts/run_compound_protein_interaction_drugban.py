#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from prettytable import PrettyTable


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


@dataclass
class FoldSpec:
    fold_id: str
    train_indices: np.ndarray
    test_indices: np.ndarray


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


def build_validation_split(train_df: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = train_df[cfg["group_column"]].fillna("MISSING").to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(cfg["val_size"]), random_state=seed)
    tr_idx, val_idx = next(splitter.split(train_df, train_df["label"].to_numpy(), groups))
    return train_df.iloc[tr_idx].copy(), train_df.iloc[val_idx].copy()


def train_rf_reference(train_df: pd.DataFrame, test_df: pd.DataFrame, compound_cols: list[str], task: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
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
    return y_score, task_test["label"].astype(int).to_numpy()


def import_drugban_modules(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from configs import get_cfg_defaults  # type: ignore
    from dataloader import DTIDataset  # type: ignore
    from models import DrugBAN, binary_cross_entropy  # type: ignore
    from trainer import Trainer  # type: ignore
    from utils import graph_collate_func, set_seed  # type: ignore

    return get_cfg_defaults, DTIDataset, DrugBAN, Trainer, graph_collate_func, set_seed, binary_cross_entropy


class EarlyStopTrainer:
    def __init__(self, base_trainer, patience: int):
        self.base = base_trainer
        self.patience = max(1, int(patience))

    def train(self):
        float2str = lambda x: "%0.4f" % x
        no_improve = 0
        self.base.best_auroc = 0
        self.base.best_model = None
        self.base.best_epoch = None
        for _ in range(self.base.epochs):
            self.base.current_epoch += 1
            train_loss = self.base.train_epoch()
            train_lst = ["epoch " + str(self.base.current_epoch)] + list(map(float2str, [train_loss]))
            self.base.train_table.add_row(train_lst)
            self.base.train_loss_epoch.append(train_loss)

            auroc, auprc, val_loss = self.base.test(dataloader="val")
            val_lst = ["epoch " + str(self.base.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.base.val_table.add_row(val_lst)
            self.base.val_loss_epoch.append(val_loss)
            self.base.val_auroc_epoch.append(auroc)

            improved = auroc >= self.base.best_auroc
            if improved:
                self.base.best_model = copy.deepcopy(self.base.model)
                self.base.best_auroc = auroc
                self.base.best_epoch = self.base.current_epoch
                no_improve = 0
            else:
                no_improve += 1

            print(
                "Validation at Epoch "
                + str(self.base.current_epoch)
                + " with validation loss "
                + str(val_loss),
                " AUROC " + str(auroc) + " AUPRC " + str(auprc),
            )
            if no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {self.base.current_epoch} with patience {self.patience}.")
                break

        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.base.test(dataloader="test")
        test_lst = ["epoch " + str(self.base.best_epoch)] + list(
            map(float2str, [auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim, test_loss])
        )
        self.base.test_table = PrettyTable(["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy", "Threshold", "Test_loss"])
        self.base.test_table.add_row(test_lst)
        print(
            "Test at Best Model of Epoch "
            + str(self.base.best_epoch)
            + " with test loss "
            + str(test_loss),
            " AUROC "
            + str(auroc)
            + " AUPRC "
            + str(auprc)
            + " Sensitivity "
            + str(sensitivity)
            + " Specificity "
            + str(specificity)
            + " Accuracy "
            + str(accuracy)
            + " Thred_optim "
            + str(thred_optim),
        )
        self.base.test_metrics["auroc"] = auroc
        self.base.test_metrics["auprc"] = auprc
        self.base.test_metrics["test_loss"] = test_loss
        self.base.test_metrics["sensitivity"] = sensitivity
        self.base.test_metrics["specificity"] = specificity
        self.base.test_metrics["accuracy"] = accuracy
        self.base.test_metrics["thred_optim"] = thred_optim
        self.base.test_metrics["best_epoch"] = self.base.best_epoch
        self.base.test_metrics["F1"] = f1
        self.base.test_metrics["Precision"] = precision
        self.base.save_result()
        return self.base.test_metrics


def predict_drugban_scores(model, loader, device, binary_cross_entropy_fn):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for v_d, v_p, y in loader:
            v_d = v_d.to(device)
            v_p = v_p.to(device)
            y = y.float().to(device)
            _v_d, _v_p, _f, score = model(v_d, v_p)
            n, _loss = binary_cross_entropy_fn(score, y)
            preds.extend(n.detach().cpu().tolist())
            labels.extend(y.detach().cpu().tolist())
    return np.asarray(preds, dtype=float), np.asarray(labels, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DrugBAN on local ESR1/AR scaffold folds.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    seed_everything(int(cfg["split"]["seed"]))

    get_cfg_defaults, DTIDataset, DrugBAN, Trainer, graph_collate_func, set_seed, binary_cross_entropy_fn = import_drugban_modules(
        Path(cfg["inputs"]["drugban_repo"])
    )
    set_seed(int(cfg["train"]["seed"]))

    pair_df = pd.read_csv(cfg["inputs"]["pairs_path"])
    target_panel = pd.read_csv(cfg["inputs"]["target_panel_path"])
    pair_df = pair_df.merge(
        target_panel[["task", "protein_sequence"]],
        on="task",
        how="left",
        validate="many_to_one",
    )
    if pair_df["protein_sequence"].isna().any():
        raise ValueError("Missing protein sequences after merging target panel.")
    tasks = list(cfg["tasks"])
    pair_df = pair_df[pair_df["task"].isin(tasks)].copy()
    folds = build_scaffold_splits(pair_df, tasks, cfg["split"])
    if not folds:
        raise ValueError("No valid scaffold folds generated.")

    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"]["device"] == "cuda" else "cpu")
    compound_cols = [c for c in pair_df.columns if c.startswith("cmp_")]

    fold_results = []
    pred_rows = []

    for fold_i, fold in enumerate(folds, start=1):
        train_df = pair_df.iloc[fold.train_indices].copy()
        test_df = pair_df.iloc[fold.test_indices].copy()
        train_core, val_df = build_validation_split(train_df, cfg["split"], int(cfg["split"]["seed"]) + fold_i)

        for df_part in (train_core, val_df, test_df):
            df_part["SMILES"] = df_part["canonical_smiles_rdkit"].astype(str)
            df_part["Protein"] = df_part["protein_sequence"].astype(str)
            df_part["Y"] = df_part["label"].astype(int)

        train_core = train_core.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_ds = DTIDataset(np.arange(len(train_core)), train_core)
        val_ds = DTIDataset(np.arange(len(val_df)), val_df)
        test_ds = DTIDataset(np.arange(len(test_df)), test_df)

        params_train = {
            "batch_size": int(cfg["train"]["batch_size"]),
            "shuffle": True,
            "num_workers": 0,
            "drop_last": True,
            "collate_fn": graph_collate_func,
        }
        params_eval = {
            "batch_size": int(cfg["train"]["batch_size"]),
            "shuffle": False,
            "num_workers": 0,
            "drop_last": False,
            "collate_fn": graph_collate_func,
        }
        train_loader = DataLoader(train_ds, **params_train)
        val_loader = DataLoader(val_ds, **params_eval)
        test_loader = DataLoader(test_ds, **params_eval)

        model_cfg = get_cfg_defaults()
        model_cfg.SOLVER.MAX_EPOCH = int(cfg["train"]["epochs"])
        model_cfg.SOLVER.BATCH_SIZE = int(cfg["train"]["batch_size"])
        model_cfg.SOLVER.LR = float(cfg["train"]["learning_rate"])
        model_cfg.SOLVER.SEED = int(cfg["train"]["seed"]) + fold_i
        model_cfg.RESULT.OUTPUT_DIR = str(out_dir / fold.fold_id)
        model_cfg.RESULT.SAVE_MODEL = False
        model_cfg.COMET.USE = False
        model_cfg.DA.TASK = False
        model_cfg.DA.USE = False

        Path(model_cfg.RESULT.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        model = DrugBAN(**model_cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.SOLVER.LR)
        trainer = Trainer(
            model=model,
            optim=optimizer,
            device=device,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            opt_da=None,
            discriminator=None,
            experiment=None,
            **model_cfg,
        )
        patience = int(cfg["train"].get("patience", 3))
        EarlyStopTrainer(trainer, patience=patience).train()

        drugban_pred, drugban_y = predict_drugban_scores(trainer.best_model, test_loader, device, binary_cross_entropy_fn)
        eval_df = test_df.reset_index(drop=True)[["pair_id", "compound_id", "task", "Y"]].copy().rename(columns={"Y": "label"})
        eval_df["fold_id"] = fold.fold_id
        eval_df["model"] = "drugban"
        eval_df["y_score"] = drugban_pred
        pred_rows.append(eval_df)

        for task in tasks:
            mask = eval_df["task"] == task
            y_true = eval_df.loc[mask, "label"].astype(int).to_numpy()
            y_score = drugban_pred[mask.to_numpy()]
            metrics = evaluate_scores(y_true, y_score)
            fold_results.append({"fold_id": fold.fold_id, "task": task, "model": "drugban", **metrics, "n_test": int(mask.sum())})

            rf_score, rf_true = train_rf_reference(train_df.rename(columns={"Y": "label"}), test_df.rename(columns={"Y": "label"}), compound_cols, task, int(cfg["train"]["seed"]) + fold_i)
            rf_eval = eval_df.loc[mask, ["pair_id", "compound_id", "task", "label"]].copy()
            rf_eval["fold_id"] = fold.fold_id
            rf_eval["model"] = "rf"
            rf_eval["y_score"] = rf_score
            pred_rows.append(rf_eval)
            metrics = evaluate_scores(rf_true, rf_score)
            fold_results.append({"fold_id": fold.fold_id, "task": task, "model": "rf", **metrics, "n_test": int(len(rf_true))})

    fold_df = pd.DataFrame(fold_results)
    agg_df = (
        fold_df.groupby(["task", "model"], as_index=False)[
            ["pr_auc", "roc_auc", "balanced_accuracy", "f1", "precision_at_5pct", "precision_at_10pct", "enrichment_factor_5pct", "n_test"]
        ]
        .mean()
        .sort_values(["task", "pr_auc"], ascending=[True, False])
    )
    preds_df = pd.concat(pred_rows, ignore_index=True)

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    agg_df.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    preds_df.to_csv(out_dir / "predictions.csv", index=False)

    status = {
        "tasks": tasks,
        "folds": len(folds),
        "device": str(device),
        "results": agg_df.to_dict(orient="records"),
    }
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# DrugBAN Environmental Panel",
        "",
        f"- device: `{device}`",
        f"- tasks: `{', '.join(tasks)}`",
        f"- folds: `{len(folds)}`",
        "",
    ]
    for task in tasks:
        lines.append(f"## {task}")
        sub = agg_df[agg_df["task"] == task]
        for _, row in sub.iterrows():
            lines.append(
                f"- `{row['model']}`: `PR-AUC {row['pr_auc']:.4f}`, `ROC-AUC {row['roc_auc']:.4f}`, `P@5% {row['precision_at_5pct']:.4f}`, `EF@5% {row['enrichment_factor_5pct']:.4f}`"
            )
        lines.append("")
    (out_dir / "status.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir / "aggregate_metrics.csv")


if __name__ == "__main__":
    main()

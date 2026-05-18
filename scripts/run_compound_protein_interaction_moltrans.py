#!/usr/bin/env python3
from __future__ import annotations

import argparse
import codecs
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from subword_nmt.apply_bpe import BPE
from torch.utils.data import DataLoader, Dataset


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


class MolTransTokenizer:
    def __init__(self, root: Path):
        protein_codes = codecs.open(root / "protein_codes_uniprot.txt", encoding="utf-8")
        self.pbpe = BPE(protein_codes, merges=-1, separator="")
        protein_map = pd.read_csv(root / "subword_units_map_uniprot.csv")
        idx2word_p = protein_map["index"].values
        self.words2idx_p = dict(zip(idx2word_p, range(len(idx2word_p))))

        drug_codes = codecs.open(root / "drug_codes_chembl.txt", encoding="utf-8")
        self.dbpe = BPE(drug_codes, merges=-1, separator="")
        drug_map = pd.read_csv(root / "subword_units_map_chembl.csv")
        idx2word_d = drug_map["index"].values
        self.words2idx_d = dict(zip(idx2word_d, range(len(idx2word_d))))

    def encode_drug(self, smiles: str, max_len: int = 50) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.dbpe.process_line(smiles).split()
        try:
            idx = np.asarray([self.words2idx_d[t] for t in tokens], dtype=np.int64)
        except Exception:
            idx = np.array([0], dtype=np.int64)
        length = len(idx)
        if length < max_len:
            arr = np.pad(idx, (0, max_len - length), constant_values=0)
            mask = np.asarray(([1] * length) + ([0] * (max_len - length)), dtype=np.int64)
        else:
            arr = idx[:max_len]
            mask = np.ones(max_len, dtype=np.int64)
        return arr, mask

    def encode_protein(self, seq: str, max_len: int = 545) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.pbpe.process_line(seq).split()
        try:
            idx = np.asarray([self.words2idx_p[t] for t in tokens], dtype=np.int64)
        except Exception:
            idx = np.array([0], dtype=np.int64)
        length = len(idx)
        if length < max_len:
            arr = np.pad(idx, (0, max_len - length), constant_values=0)
            mask = np.asarray(([1] * length) + ([0] * (max_len - length)), dtype=np.int64)
        else:
            arr = idx[:max_len]
            mask = np.ones(max_len, dtype=np.int64)
        return arr, mask


class MolTransDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: MolTransTokenizer):
        self.smiles = df["canonical_smiles_rdkit"].astype(str).tolist()
        self.protein = df["protein_sequence"].astype(str).tolist()
        self.labels = df["label"].astype(int).to_numpy()
        self.compounds = df["compound_id"].astype(str).tolist()
        self.tasks = df["task"].astype(str).tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        d, d_mask = self.tokenizer.encode_drug(self.smiles[idx])
        p, p_mask = self.tokenizer.encode_protein(self.protein[idx])
        return (
            torch.tensor(d, dtype=torch.long),
            torch.tensor(p, dtype=torch.long),
            torch.tensor(d_mask, dtype=torch.long),
            torch.tensor(p_mask, dtype=torch.long),
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        )


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, variance_epsilon: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_size: int, dropout_rate: float):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(position_ids)
        embeddings = words + positions
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_dropout: float):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        return context.view(*new_shape)


class SelfOutput(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_dropout: float, hidden_dropout: float):
        super().__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_dropout)
        self.output = SelfOutput(hidden_size, hidden_dropout)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        return self.output(self_output, input_tensor)


class Intermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return F.relu(self.dense(hidden_states))


class Output(nn.Module):
    def __init__(self, intermediate_size: int, hidden_size: int, hidden_dropout: float):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_attention_heads: int, attention_dropout: float, hidden_dropout: float):
        super().__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_dropout, hidden_dropout)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout)

    def forward(self, hidden_states, attention_mask):
        attn = self.attention(hidden_states, attention_mask)
        intermediate = self.intermediate(attn)
        return self.output(intermediate, attn)


class EncoderMultipleLayers(nn.Module):
    def __init__(self, n_layer: int, hidden_size: int, intermediate_size: int, num_attention_heads: int, attention_dropout: float, hidden_dropout: float):
        super().__init__()
        layer = EncoderLayer(hidden_size, intermediate_size, num_attention_heads, attention_dropout, hidden_dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class MolTransInteraction(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.max_d = int(config["max_drug_seq"])
        self.max_p = int(config["max_protein_seq"])
        self.emb_size = int(config["emb_size"])
        self.dropout_rate = float(config["dropout_rate"])
        self.input_dim_drug = int(config["input_dim_drug"])
        self.input_dim_target = int(config["input_dim_target"])
        self.n_layer = int(config["n_layer"])
        self.hidden_size = int(config["emb_size"])
        self.intermediate_size = int(config["intermediate_size"])
        self.num_attention_heads = int(config["num_attention_heads"])
        self.attention_probs_dropout_prob = float(config["attention_probs_dropout_prob"])
        self.hidden_dropout_prob = float(config["hidden_dropout_prob"])

        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)
        self.d_encoder = EncoderMultipleLayers(
            self.n_layer,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.attention_probs_dropout_prob,
            self.hidden_dropout_prob,
        )
        self.p_encoder = EncoderMultipleLayers(
            self.n_layer,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.attention_probs_dropout_prob,
            self.hidden_dropout_prob,
        )
        self.icnn = nn.Conv2d(1, 3, 3, padding=0)
        flat_dim = 3 * (self.max_d - 2) * (self.max_p - 2)
        self.decoder = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )

    def forward(self, d, p, d_mask, p_mask):
        ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2)
        ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0

        d_emb = self.demb(d)
        p_emb = self.pemb(p)
        d_encoded = self.d_encoder(d_emb.float(), ex_d_mask.float())
        p_encoded = self.p_encoder(p_emb.float(), ex_p_mask.float())

        d_aug = torch.unsqueeze(d_encoded, 2).repeat(1, 1, self.max_p, 1)
        p_aug = torch.unsqueeze(p_encoded, 1).repeat(1, self.max_d, 1, 1)
        interaction = d_aug * p_aug
        batch_size = d.size(0)
        interaction = interaction.view(batch_size, -1, self.max_d, self.max_p).sum(dim=1, keepdim=True)
        interaction = F.dropout(interaction, p=self.dropout_rate, training=self.training)
        feat = self.icnn(interaction).view(batch_size, -1)
        return self.decoder(feat).squeeze(-1)


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


def build_validation_split(train_df: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = train_df[cfg["group_column"]].fillna("MISSING").to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(cfg["val_size"]), random_state=seed)
    tr_idx, val_idx = next(splitter.split(train_df, train_df["label"].to_numpy(), groups))
    return train_df.iloc[tr_idx].copy(), train_df.iloc[val_idx].copy()


def predict_model(model: MolTransInteraction, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for d, p, d_mask, p_mask, y in loader:
            d = d.to(device)
            p = p.to(device)
            d_mask = d_mask.to(device)
            p_mask = p_mask.to(device)
            logits = model(d, p, d_mask, p_mask)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def train_moltrans(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: MolTransTokenizer,
    cfg: dict,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    seed_everything(seed)
    train_core, val_df = build_validation_split(train_df, cfg["split"], seed + 11)
    train_ds = MolTransDataset(train_core, tokenizer)
    val_ds = MolTransDataset(val_df, tokenizer)
    test_ds = MolTransDataset(test_df, tokenizer)

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MolTransInteraction(cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["learning_rate"]), weight_decay=float(cfg["train"]["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val = -1.0
    patience = int(cfg["train"]["patience"])
    patience_left = patience
    for _epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        for d, p, d_mask, p_mask, y in train_loader:
            d = d.to(device)
            p = p.to(device)
            d_mask = d_mask.to(device)
            p_mask = p_mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(d, p, d_mask, p_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_pred, val_y = predict_model(model, val_loader, device)
        val_pr = safe_metric(average_precision_score, val_y, val_pred)
        if val_pr > best_val:
            best_val = val_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return predict_model(model, test_loader, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MolTrans on the environmental ESR1/AR panel using local scaffold splits.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    seed_everything(int(cfg["split"]["seed"]))
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

    tokenizer = MolTransTokenizer(Path(cfg["inputs"]["espf_dir"]))
    compound_cols = [c for c in pair_df.columns if c.startswith("cmp_")]
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"]["device"] == "cuda" else "cpu")

    results = []
    pred_rows = []
    for fold in folds:
        train_df = pair_df.iloc[fold.train_indices].copy()
        test_df = pair_df.iloc[fold.test_indices].copy()

        moltrans_pred, moltrans_y = train_moltrans(train_df, test_df, tokenizer, cfg, int(cfg["split"]["seed"]), device)
        moltrans_eval_df = test_df[["pair_id", "compound_id", "task", "label"]].copy()
        moltrans_eval_df["fold_id"] = fold.fold_id
        moltrans_eval_df["model"] = "moltrans"
        moltrans_eval_df["y_score"] = moltrans_pred
        pred_rows.append(moltrans_eval_df)

        for task in tasks:
            mask = test_df["task"] == task
            y_true = test_df.loc[mask, "label"].astype(int).to_numpy()
            y_score = moltrans_pred[mask.to_numpy()]
            metrics = evaluate_scores(y_true, y_score)
            results.append({"fold_id": fold.fold_id, "task": task, "model": "moltrans", **metrics, "n_test": int(mask.sum())})

            rf_score, rf_true, rf_compounds = train_rf_reference(train_df, test_df, compound_cols, task, int(cfg["split"]["seed"]))
            rf_rows = pd.DataFrame(
                {
                    "pair_id": test_df.loc[mask, "pair_id"].astype(str).tolist(),
                    "compound_id": rf_compounds,
                    "task": task,
                    "label": rf_true,
                    "fold_id": fold.fold_id,
                    "model": "rf",
                    "y_score": rf_score,
                }
            )
            pred_rows.append(rf_rows)
            metrics = evaluate_scores(rf_true, rf_score)
            results.append({"fold_id": fold.fold_id, "task": task, "model": "rf", **metrics, "n_test": int(len(rf_true))})

    results_df = pd.DataFrame(results)
    summary_df = (
        results_df.groupby(["task", "model"], as_index=False)[
            ["pr_auc", "roc_auc", "balanced_accuracy", "f1", "precision_at_5pct", "precision_at_10pct", "enrichment_factor_5pct", "n_test"]
        ]
        .mean()
        .sort_values(["task", "pr_auc"], ascending=[True, False])
    )
    pred_df = pd.concat(pred_rows, ignore_index=True)

    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    lines = [
        "# MolTrans Environmental Panel",
        "",
        f"- device: `{device}`",
        f"- tasks: `{', '.join(tasks)}`",
        f"- folds: `{len(folds)}`",
        "",
    ]
    for task in tasks:
        lines.append(f"## {task}")
        sub = summary_df[summary_df["task"] == task]
        for _, row in sub.iterrows():
            lines.append(
                f"- `{row['model']}`: `PR-AUC {row['pr_auc']:.4f}`, `ROC-AUC {row['roc_auc']:.4f}`, `P@5% {row['precision_at_5pct']:.4f}`, `EF@5% {row['enrichment_factor_5pct']:.4f}`"
            )
        lines.append("")
    (out_dir / "status.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir / "aggregate_metrics.csv")


if __name__ == "__main__":
    main()

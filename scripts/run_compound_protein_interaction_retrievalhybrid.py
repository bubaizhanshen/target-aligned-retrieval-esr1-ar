#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}


@dataclass
class FoldSpec:
    fold_id: str
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class MemoryTaskCache:
    memory_name: str
    task: str
    memory_indices: np.ndarray
    memory_bits_t: np.ndarray
    memory_bit_sums: np.ndarray
    memory_labels: np.ndarray
    local_pos_by_global: np.ndarray
    exclude_self: bool


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


def pairwise_ranking_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    diffs = pos[:, None] - neg[None, :]
    return torch.nn.functional.softplus(-diffs).mean()


def tokenize_protein_sequence(seq: str, chunk_size: int, max_chunks: int) -> np.ndarray:
    chunks = []
    seq = (seq or "").strip().upper()
    for i in range(0, min(len(seq), chunk_size * max_chunks), chunk_size):
        chunk = seq[i : i + chunk_size]
        ids = [AA_TO_IDX.get(aa, 0) for aa in chunk]
        if len(ids) < chunk_size:
            ids.extend([0] * (chunk_size - len(ids)))
        chunks.append(ids)
    while len(chunks) < max_chunks:
        chunks.append([0] * chunk_size)
    return np.asarray(chunks[:max_chunks], dtype=np.int64)


def build_target_sequence_tables(target_panel_path: Path, chunk_size: int, max_chunks: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    panel = pd.read_csv(target_panel_path)
    seq_map = {}
    target_id_map = {}
    for i, row in panel.iterrows():
        accession = str(row["uniprot_accession"])
        seq_map[accession] = tokenize_protein_sequence(str(row["protein_sequence"]), chunk_size, max_chunks)
        target_id_map[accession] = i
    return seq_map, target_id_map


def memory_mask(bank: pd.DataFrame, memory_name: str) -> np.ndarray:
    if memory_name == "pollutant_only":
        return bank["memory_eligible_pollutant_only"].fillna(0).astype(int).to_numpy().astype(bool)
    if memory_name == "generic_bioactive":
        return bank["memory_eligible_generic_bioactive"].fillna(0).astype(int).to_numpy().astype(bool)
    if memory_name == "mixed":
        return bank["memory_eligible_mixed"].fillna(0).astype(int).to_numpy().astype(bool)
    column_name = f"memory_eligible_{memory_name}"
    if column_name in bank.columns:
        return bank[column_name].fillna(0).astype(int).to_numpy().astype(bool)
    raise ValueError(f"Unsupported memory: {memory_name}")


def build_memory_task_cache(
    bank: pd.DataFrame,
    fp_matrix: np.ndarray,
    task: str,
    memory_name: str,
    *,
    label_shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> MemoryTaskCache:
    memory_base = memory_mask(bank, memory_name)
    memory_task_mask = memory_base & bank[task].notna().to_numpy()
    memory_indices = np.where(memory_task_mask)[0]
    memory_bits = fp_matrix[memory_indices].astype(np.float32, copy=False)
    memory_bits_t = memory_bits.T.copy()
    memory_bit_sums = memory_bits.sum(axis=1).astype(np.float32, copy=False)
    memory_labels = bank.iloc[memory_indices][task].astype(int).to_numpy()
    if label_shuffle and len(memory_labels) > 1:
        rng_local = rng if rng is not None else np.random.default_rng(0)
        memory_labels = memory_labels.copy()
        rng_local.shuffle(memory_labels)
    exclude_self = bool(bank.iloc[memory_indices]["memory_domain"].astype(str).eq("pollutant").any())
    local_pos_by_global = np.full(len(bank), -1, dtype=int)
    local_pos_by_global[memory_indices] = np.arange(len(memory_indices))
    return MemoryTaskCache(
        memory_name=memory_name,
        task=task,
        memory_indices=memory_indices,
        memory_bits_t=memory_bits_t,
        memory_bit_sums=memory_bit_sums,
        memory_labels=memory_labels,
        local_pos_by_global=local_pos_by_global,
        exclude_self=exclude_self,
    )


def compute_retrieval_stats_batch(
    fp_matrix: np.ndarray,
    query_globals: np.ndarray,
    cache: MemoryTaskCache,
    top_k: int,
    exclude_self: bool,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    if len(cache.memory_indices) == 0:
        return np.empty((0, 7), dtype=np.float32), np.zeros(len(query_globals), dtype=bool)
    available = len(cache.memory_indices) - (1 if exclude_self else 0)
    if available < top_k:
        return np.empty((0, 7), dtype=np.float32), np.zeros(len(query_globals), dtype=bool)

    all_stats: list[np.ndarray] = []
    valid_mask_parts: list[np.ndarray] = []
    for start in range(0, len(query_globals), chunk_size):
        chunk_globals = query_globals[start : start + chunk_size]
        q_bits = fp_matrix[chunk_globals].astype(np.float32, copy=False)
        intersections = q_bits @ cache.memory_bits_t
        q_sums = q_bits.sum(axis=1, keepdims=True)
        unions = cache.memory_bit_sums[None, :] + q_sums - intersections
        sims = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=np.float32), where=unions > 0)
        valid_chunk = np.ones(len(chunk_globals), dtype=bool)
        if exclude_self:
            self_local = cache.local_pos_by_global[chunk_globals]
            has_self = self_local >= 0
            valid_chunk &= has_self
            if has_self.any():
                sims[np.arange(len(chunk_globals))[has_self], self_local[has_self]] = -1.0
        valid_mask_parts.append(valid_chunk)
        top_local = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
        top_scores = np.take_along_axis(sims, top_local, axis=1)
        sort_order = np.argsort(top_scores, axis=1)[:, ::-1]
        top_local = np.take_along_axis(top_local, sort_order, axis=1)
        top_scores = np.take_along_axis(top_scores, sort_order, axis=1)
        top_labels = cache.memory_labels[top_local].astype(np.float32)
        weight_sum = top_scores.sum(axis=1)
        knn_score = np.divide((top_scores * top_labels).sum(axis=1), weight_sum, out=top_labels.mean(axis=1), where=weight_sum > 0)
        pos_mask = top_labels == 1
        neg_mask = top_labels == 0
        pos_max = np.where(pos_mask.any(axis=1), np.where(pos_mask, top_scores, -1.0).max(axis=1), 0.0)
        neg_max = np.where(neg_mask.any(axis=1), np.where(neg_mask, top_scores, -1.0).max(axis=1), 0.0)
        stats = np.column_stack(
            [knn_score, top_scores[:, 0], top_scores.mean(axis=1), top_labels.mean(axis=1), pos_max, neg_max, pos_max - neg_max]
        ).astype(np.float32)
        all_stats.append(stats)
    return np.vstack(all_stats), np.concatenate(valid_mask_parts)


def build_retrieval_feature_table(
    pair_df: pd.DataFrame,
    bank: pd.DataFrame,
    fp_matrix_bank: np.ndarray,
    memories: list[str],
    top_k: int,
    *,
    control_mode: str = "none",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    control_mode = str(control_mode or "none")
    if control_mode not in {"none", "random_memory", "label_shuffled"}:
        raise ValueError(f"Unsupported retrieval control_mode: {control_mode}")
    compound_to_global = {cid: i for i, cid in enumerate(bank["compound_id"].astype(str).tolist())}
    query_globals = pair_df["compound_id"].astype(str).map(compound_to_global).to_numpy()
    if np.any(pd.isna(query_globals)):
        missing = pair_df.loc[pd.isna(query_globals), "compound_id"].astype(str).unique()[:5]
        raise ValueError(f"Missing retrieval bank compounds: {missing.tolist()}")
    query_globals = query_globals.astype(int)

    retrieval_blocks = []
    for memory_name in memories:
        rows = []
        for task in pair_df["task"].astype(str).unique():
            task_mask = pair_df["task"].astype(str).to_numpy() == task
            cache = build_memory_task_cache(
                bank,
                fp_matrix_bank,
                task,
                memory_name,
                label_shuffle=(control_mode == "label_shuffled"),
                rng=rng,
            )
            stats, valid_mask = compute_retrieval_stats_batch(
                fp_matrix=fp_matrix_bank,
                query_globals=query_globals[task_mask],
                cache=cache,
                top_k=top_k,
                exclude_self=cache.exclude_self,
            )
            task_rows = np.zeros((task_mask.sum(), 7), dtype=np.float32)
            if len(stats) > 0 and valid_mask.any():
                task_rows[valid_mask] = stats[valid_mask]
            if control_mode == "random_memory" and len(task_rows) > 1:
                perm = (rng if rng is not None else np.random.default_rng(0)).permutation(len(task_rows))
                task_rows = task_rows[perm]
            rows.append((task_mask, task_rows))
        block = np.zeros((len(pair_df), 7), dtype=np.float32)
        for task_mask, task_rows in rows:
            block[task_mask] = task_rows
        retrieval_blocks.append(block)
    return np.concatenate(retrieval_blocks, axis=1).astype(np.float32)


class PairDataset(Dataset):
    def __init__(self, descriptor: np.ndarray, fp_chunks: np.ndarray, prot_chunks: np.ndarray, target_ids: np.ndarray, retrieval_feats: np.ndarray, labels: np.ndarray):
        self.descriptor = torch.tensor(descriptor, dtype=torch.float32)
        self.fp_chunks = torch.tensor(fp_chunks, dtype=torch.float32)
        self.prot_chunks = torch.tensor(prot_chunks, dtype=torch.long)
        self.target_ids = torch.tensor(target_ids, dtype=torch.long)
        self.retrieval_feats = torch.tensor(retrieval_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.descriptor[idx], self.fp_chunks[idx], self.prot_chunks[idx], self.target_ids[idx], self.retrieval_feats[idx], self.labels[idx]


class RetrievalAugmentedSeqCPI(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        chunk_size: int,
        max_chunks: int,
        num_targets: int,
        retrieval_dim: int,
        num_memories: int,
        residue_embed_dim: int = 16,
        memory_fusion: str = "concat",
    ):
        super().__init__()
        self.desc_proj = nn.Linear(8, hidden_dim)
        self.fp_proj = nn.Linear(32, hidden_dim)
        self.residue_embedding = nn.Embedding(len(AA_TO_IDX) + 1, residue_embed_dim, padding_idx=0)
        self.prot_chunk_proj = nn.Linear(residue_embed_dim, hidden_dim)
        self.target_embedding = nn.Embedding(num_targets, hidden_dim)
        self.position_embedding = nn.Embedding(max_chunks, hidden_dim)
        self.comp_ln = nn.LayerNorm(hidden_dim)
        self.prot_ln = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.memory_fusion = memory_fusion
        self.num_memories = num_memories
        if memory_fusion == "gated_moe":
            self.memory_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(retrieval_dim // num_memories, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                    )
                    for _ in range(num_memories)
                ]
            )
            self.memory_gate = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_memories),
            )
        else:
            self.retrieval_branch = nn.Sequential(
                nn.Linear(retrieval_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.max_chunks = max_chunks

    def encode_protein(self, prot_chunks: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        residue_emb = self.residue_embedding(prot_chunks)
        mask = (prot_chunks != 0).unsqueeze(-1)
        counts = mask.sum(dim=2).clamp(min=1)
        chunk_emb = (residue_emb * mask).sum(dim=2) / counts
        chunk_tokens = self.prot_chunk_proj(chunk_emb)
        positions = torch.arange(self.max_chunks, device=prot_chunks.device).unsqueeze(0)
        chunk_tokens = chunk_tokens + self.position_embedding(positions)
        target_token = self.target_embedding(target_ids).unsqueeze(1)
        return self.prot_ln(torch.cat([target_token, chunk_tokens], dim=1))

    def forward(self, descriptor, fp_chunks, prot_chunks, target_ids, retrieval_feats):
        desc_token = self.desc_proj(descriptor).unsqueeze(1)
        fp_tokens = self.fp_proj(fp_chunks)
        comp_tokens = self.comp_ln(torch.cat([desc_token, fp_tokens], dim=1))
        prot_tokens = self.encode_protein(prot_chunks, target_ids)

        comp_ctx, _ = self.cross_attn(comp_tokens, prot_tokens, prot_tokens, need_weights=False)
        prot_ctx, _ = self.cross_attn(prot_tokens, comp_tokens, comp_tokens, need_weights=False)
        comp_ctx, _ = self.self_attn(comp_ctx, comp_ctx, comp_ctx, need_weights=False)
        prot_ctx, _ = self.self_attn(prot_ctx, prot_ctx, prot_ctx, need_weights=False)

        comp_pool = comp_ctx.mean(dim=1)
        prot_pool = prot_ctx.mean(dim=1)
        prod = comp_pool * prot_pool
        diff = torch.abs(comp_pool - prot_pool)
        if self.memory_fusion == "gated_moe":
            memory_chunks = retrieval_feats.view(retrieval_feats.size(0), self.num_memories, -1)
            expert_outs = [expert(memory_chunks[:, i, :]) for i, expert in enumerate(self.memory_experts)]
            expert_stack = torch.stack(expert_outs, dim=1)
            gate_in = torch.cat([comp_pool, prot_pool, prod, diff], dim=1)
            gate = torch.softmax(self.memory_gate(gate_in), dim=1).unsqueeze(-1)
            retrieval_vec = (expert_stack * gate).sum(dim=1)
        else:
            retrieval_vec = self.retrieval_branch(retrieval_feats)
        return self.head(torch.cat([comp_pool, prot_pool, prod, diff, retrieval_vec], dim=1)).squeeze(-1)


def train_rf_reference(train_df: pd.DataFrame, test_df: pd.DataFrame, compound_cols: list[str], task: str, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    task_train = train_df[train_df["task"] == task].copy()
    task_test = test_df[test_df["task"] == task].copy()
    model = RandomForestClassifier(n_estimators=120, class_weight="balanced", random_state=seed, n_jobs=-1)
    model.fit(task_train[compound_cols].to_numpy(dtype=float), task_train["label"].astype(int).to_numpy())
    y_score = model.predict_proba(task_test[compound_cols].to_numpy(dtype=float))[:, 1]
    return y_score, task_test["label"].astype(int).to_numpy(), task_test["compound_id"].astype(str).tolist()


def build_arrays(df: pd.DataFrame, seq_map: dict[str, np.ndarray], target_id_map: dict[str, int]):
    desc_cols = [c for c in df.columns if c.startswith("cmp_") and not c.startswith("cmp_fp_")]
    fp_cols = [c for c in df.columns if c.startswith("cmp_fp_")]
    desc = df[desc_cols].to_numpy(dtype=float)
    fp = df[fp_cols].to_numpy(dtype=float)
    prot_chunks = np.stack([seq_map[str(acc)] for acc in df["uniprot_accession"].astype(str)], axis=0)
    target_ids = np.asarray([target_id_map[str(acc)] for acc in df["uniprot_accession"].astype(str)], dtype=np.int64)
    return desc, fp, prot_chunks, target_ids, desc_cols


def reshape_fp(fp: np.ndarray) -> np.ndarray:
    return fp.reshape(len(fp), 16, 32)


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, retrieval_train: np.ndarray, retrieval_val: np.ndarray, cfg: dict, seed: int, seq_map: dict[str, np.ndarray], target_id_map: dict[str, int]):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    desc_tr, fp_tr, prot_tr, target_ids_tr, _ = build_arrays(train_df, seq_map, target_id_map)
    desc_val, fp_val, prot_val, target_ids_val, _ = build_arrays(val_df, seq_map, target_id_map)
    desc_mean, desc_std = standardize_fit(desc_tr)
    ret_mean, ret_std = standardize_fit(retrieval_train)
    desc_tr = standardize_apply(desc_tr, desc_mean, desc_std)
    desc_val = standardize_apply(desc_val, desc_mean, desc_std)
    retrieval_train = standardize_apply(retrieval_train, ret_mean, ret_std)
    retrieval_val = standardize_apply(retrieval_val, ret_mean, ret_std)
    y_tr = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()

    train_dataset = PairDataset(desc_tr, reshape_fp(fp_tr), prot_tr, target_ids_tr, retrieval_train, y_tr)
    val_dataset = PairDataset(desc_val, reshape_fp(fp_val), prot_val, target_ids_val, retrieval_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model = RetrievalAugmentedSeqCPI(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        chunk_size=int(cfg["protein_encoder"]["chunk_size"]),
        max_chunks=int(cfg["protein_encoder"]["max_chunks"]),
        num_targets=len(target_id_map),
        retrieval_dim=retrieval_train.shape[1],
        num_memories=len(cfg["retrieval"]["memories"]),
        residue_embed_dim=int(cfg["protein_encoder"]["residue_embed_dim"]),
        memory_fusion=str(cfg["model"].get("memory_fusion", "concat")),
    ).to(device)

    pos = max(1.0, float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum())))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    rank_loss_weight = float(cfg["training"].get("rank_loss_weight", 0.0))

    best_state = None
    best_pr = -np.inf
    no_improve = 0
    for _epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        for desc, fp, prot_chunks, target_ids, retrieval_feats, y in train_loader:
            desc, fp, prot_chunks, target_ids, retrieval_feats, y = [x.to(device) for x in (desc, fp, prot_chunks, target_ids, retrieval_feats, y)]
            optimizer.zero_grad()
            logits = model(desc, fp, prot_chunks, target_ids, retrieval_feats)
            loss = criterion(logits, y)
            if rank_loss_weight > 0:
                loss = loss + rank_loss_weight * pairwise_ranking_loss(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_scores = []
        with torch.no_grad():
            for desc, fp, prot_chunks, target_ids, retrieval_feats, _y in val_loader:
                desc, fp, prot_chunks, target_ids, retrieval_feats = [x.to(device) for x in (desc, fp, prot_chunks, target_ids, retrieval_feats)]
                val_scores.append(torch.sigmoid(model(desc, fp, prot_chunks, target_ids, retrieval_feats)).cpu().numpy())
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
    norms = {"desc_mean": desc_mean, "desc_std": desc_std, "ret_mean": ret_mean, "ret_std": ret_std}
    return model, norms


def predict_model(model, norms, test_df: pd.DataFrame, retrieval_test: np.ndarray, seq_map: dict[str, np.ndarray], target_id_map: dict[str, int]) -> np.ndarray:
    desc, fp, prot_chunks, target_ids, _ = build_arrays(test_df, seq_map, target_id_map)
    desc = standardize_apply(desc, norms["desc_mean"], norms["desc_std"])
    retrieval_test = standardize_apply(retrieval_test, norms["ret_mean"], norms["ret_std"])
    dataset = PairDataset(desc, reshape_fp(fp), prot_chunks, target_ids, retrieval_test, test_df["label"].astype(int).to_numpy())
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    scores = []
    with torch.no_grad():
        for desc, fp, prot_chunks, target_ids, retrieval_feats, _y in loader:
            scores.append(torch.sigmoid(model(desc, fp, prot_chunks, target_ids, retrieval_feats)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(len(test_df), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-augmented sequence CPI model.")
    parser.add_argument("--config", default="configs/compound_protein_interaction_retrievalhybrid_v1.yaml")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--results-dir", default="")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    pair_dir = Path(cfg["inputs"]["pair_dataset_dir"])
    target_panel_path = Path(cfg["inputs"]["target_panel_path"])
    bank_dir = Path(cfg["inputs"]["retrieval_bank_dir"])
    results_dir = Path(args.results_dir) if args.results_dir.strip() else Path(cfg["outputs"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    seq_map, target_id_map = build_target_sequence_tables(
        target_panel_path,
        chunk_size=int(cfg["protein_encoder"]["chunk_size"]),
        max_chunks=int(cfg["protein_encoder"]["max_chunks"]),
    )

    df = pd.read_csv(pair_dir / "pair_features.csv", low_memory=False)
    tasks = list(cfg["tasks"]["core"])
    df = df[df["task"].isin(tasks)].copy()
    df["label"] = df["label"].astype(int)

    bank = pd.read_csv(bank_dir / "retrieval_bank.csv", low_memory=False)
    fp_cols_bank = sorted(c for c in bank.columns if c.startswith("fp_"))
    fp_matrix_bank = bank[fp_cols_bank].fillna(0.0).to_numpy(dtype=np.uint8)
    seed = int(cfg["training"]["random_seed"])
    retrieval_features = build_retrieval_feature_table(
        pair_df=df,
        bank=bank,
        fp_matrix_bank=fp_matrix_bank,
        memories=list(cfg["retrieval"]["memories"]),
        top_k=int(cfg["retrieval"]["top_k"]),
        control_mode=str(cfg["retrieval"].get("control_mode", "none")),
        rng=np.random.default_rng(seed),
    )

    control_mode = str(cfg["retrieval"].get("control_mode", "none"))
    model_name = "retrieval_augmented_seq_cpi"
    if control_mode == "random_memory":
        model_name = "retrieval_augmented_seq_cpi_random_memory"
    elif control_mode == "label_shuffled":
        model_name = "retrieval_augmented_seq_cpi_label_shuffled"

    compound_cols = sorted(c for c in df.columns if c.startswith("cmp_"))
    folds = build_scaffold_splits(df, tasks, cfg["training"]["split"])
    if args.max_folds and args.max_folds > 0:
        folds = folds[: args.max_folds]
    metric_rows = []
    pred_rows = []

    for fold_idx, fold in enumerate(folds, start=1):
        train_df = df.iloc[fold.train_indices].reset_index(drop=True)
        test_df = df.iloc[fold.test_indices].reset_index(drop=True)
        retrieval_train_all = retrieval_features[fold.train_indices]
        retrieval_test_all = retrieval_features[fold.test_indices]

        for task in tasks:
            y_score, y_true, comp_ids = train_rf_reference(train_df, test_df, compound_cols, task, seed + fold_idx)
            metrics = evaluate_scores(y_true, y_score)
            metric_rows.append({"model": "rf_reference_single_task", "task": task, "split_type": "scaffold_holdout", "fold_id": fold.fold_id, **metrics})
            for cid, truth, score in zip(comp_ids, y_true, y_score):
                pred_rows.append({"model": "rf_reference_single_task", "task": task, "split_type": "scaffold_holdout", "fold_id": fold.fold_id, "compound_id": cid, "y_true": int(truth), "y_score": float(score)})

        rng = np.random.default_rng(seed + fold_idx)
        idx = np.arange(len(train_df))
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * 0.85)))
        tr_idx, val_idx = idx[:cut], idx[cut:]
        train_sub = train_df.iloc[tr_idx].reset_index(drop=True)
        val_sub = train_df.iloc[val_idx].reset_index(drop=True)
        retrieval_train = retrieval_train_all[tr_idx]
        retrieval_val = retrieval_train_all[val_idx]

        model, norms = train_model(train_sub, val_sub, retrieval_train, retrieval_val, cfg, seed + fold_idx, seq_map, target_id_map)
        y_score_all = predict_model(model, norms, test_df, retrieval_test_all, seq_map, target_id_map)
        test_pred = test_df[["compound_id", "task", "label"]].copy()
        test_pred["y_score"] = y_score_all
        for task in tasks:
            task_df = test_pred[test_pred["task"] == task].copy()
            y_true = task_df["label"].astype(int).to_numpy()
            y_score = task_df["y_score"].astype(float).to_numpy()
            metrics = evaluate_scores(y_true, y_score)
            metric_rows.append({"model": model_name, "task": task, "split_type": "scaffold_holdout", "fold_id": fold.fold_id, **metrics})
            for _, row in task_df.iterrows():
                pred_rows.append({"model": model_name, "task": task, "split_type": "scaffold_holdout", "fold_id": fold.fold_id, "compound_id": row["compound_id"], "y_true": int(row["label"]), "y_score": float(row["y_score"])})

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
    best = summary.sort_values(["task", "pr_auc_mean"], ascending=[True, False]).groupby("task", as_index=False).head(1).reset_index(drop=True)
    best.to_csv(results_dir / "best_by_task.csv", index=False)
    lines = [
        "# Retrieval-Augmented Sequence CPI v1",
        f"- retrieval_control_mode: `{control_mode}`",
        "",
        f"- folds: `{len(folds)}`",
        f"- tasks: `{', '.join(tasks)}`",
        f"- memories: `{', '.join(cfg['retrieval']['memories'])}`",
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

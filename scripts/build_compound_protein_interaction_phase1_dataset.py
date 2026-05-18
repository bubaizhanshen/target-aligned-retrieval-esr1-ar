#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def hash_index(token: str, n_bins: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_bins


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return counts.astype(float)
    return counts.astype(float) / float(total)


def build_protein_feature_rows(target_panel: pd.DataFrame, hash_bins: int) -> pd.DataFrame:
    family_values = sorted(target_panel["target_family"].dropna().unique().tolist())
    rows: list[dict] = []
    for _, row in target_panel.iterrows():
        seq = str(row["protein_sequence"])
        feature_row = {
            "task": row["task"],
            "gene_symbol": row["gene_symbol"],
            "uniprot_accession": row["uniprot_accession"],
            "target_role": row["role"],
            "target_family": row["target_family"],
            "prot_seq_len": len(seq),
            "prot_seq_len_log1p": np.log1p(len(seq)),
        }

        aa_counts = np.zeros(len(AMINO_ACIDS), dtype=float)
        aa_index = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
        for aa in seq:
            if aa in aa_index:
                aa_counts[aa_index[aa]] += 1.0
        aa_comp = normalize_counts(aa_counts)
        for aa, value in zip(AMINO_ACIDS, aa_comp):
            feature_row[f"prot_aa_{aa}"] = float(value)

        di_counts = np.zeros(hash_bins, dtype=float)
        for i in range(len(seq) - 1):
            token = seq[i : i + 2]
            di_counts[hash_index(token, hash_bins)] += 1.0
        di_freq = normalize_counts(di_counts)
        for idx, value in enumerate(di_freq):
            feature_row[f"prot_kmer2hash_{idx:03d}"] = float(value)

        for family in family_values:
            feature_row[f"prot_family__{family}"] = 1.0 if row["target_family"] == family else 0.0

        rows.append(feature_row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1 compound-protein interaction pair dataset.")
    parser.add_argument(
        "--config",
        default="configs/compound_protein_interaction_v1.yaml",
        help="Path to the compound-protein interaction YAML config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    compound_dir = Path(cfg["inputs"]["compound_dataset_dir"])
    target_panel_csv = Path(cfg["inputs"]["target_panel_csv"])
    out_dir = Path(cfg["inputs"]["pair_dataset_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    compounds = pd.read_csv(compound_dir / "compounds.csv", low_memory=False)
    compound_features = pd.read_csv(compound_dir / "compound_features.csv", low_memory=False)
    target_panel = pd.read_csv(target_panel_csv, low_memory=False)

    core_tasks = list(cfg["tasks"]["core"])
    target_panel = target_panel[target_panel["task"].isin(core_tasks)].copy()
    if target_panel.empty:
        raise ValueError("No core tasks from config found in target panel.")

    hash_bins = int(cfg["protein_encoder"].get("hash_bins", 128))
    protein_features = build_protein_feature_rows(target_panel, hash_bins=hash_bins)

    compound_feature_cols = [c for c in compound_features.columns if c != "compound_id"]
    feature_map = compound_features.rename(columns={c: f"cmp_{c}" for c in compound_feature_cols})
    compounds = compounds.merge(feature_map, on="compound_id", how="inner")

    pair_rows: list[pd.DataFrame] = []
    for _, target in target_panel.iterrows():
        task = target["task"]
        if task not in compounds.columns:
            continue
        task_df = compounds[compounds[task].notna()].copy()
        if task_df.empty:
            continue
        task_df["task"] = task
        task_df["label"] = task_df[task].astype(int)
        task_df["pair_id"] = task_df["compound_id"].astype(str) + "::" + task
        task_df["gene_symbol"] = target["gene_symbol"]
        task_df["uniprot_accession"] = target["uniprot_accession"]
        task_df["target_role"] = target["role"]
        task_df["target_family"] = target["target_family"]
        pair_rows.append(task_df)

    if not pair_rows:
        raise ValueError("No labeled compound-target pairs were generated.")

    pairs = pd.concat(pair_rows, ignore_index=True)
    pairs = pairs.merge(
        protein_features,
        on=["task", "gene_symbol", "uniprot_accession", "target_role", "target_family"],
        how="left",
    )

    pair_meta_cols = [
        "pair_id",
        "compound_id",
        "task",
        "label",
        "gene_symbol",
        "uniprot_accession",
        "target_role",
        "target_family",
        "murcko_scaffold",
        "domain",
        "dtxsid",
        "name",
        "casrn",
        "chemical_class",
        "canonical_smiles_rdkit",
    ]
    pair_meta_cols = [c for c in pair_meta_cols if c in pairs.columns]
    cmp_cols = sorted(c for c in pairs.columns if c.startswith("cmp_"))
    prot_cols = sorted(c for c in pairs.columns if c.startswith("prot_"))

    pair_features = pairs[pair_meta_cols + cmp_cols + prot_cols].copy()
    pair_features.to_csv(out_dir / "pair_features.csv", index=False)
    pairs[pair_meta_cols].to_csv(out_dir / "pairs.csv", index=False)
    protein_features.to_csv(out_dir / "protein_features.csv", index=False)

    summary = {
        "study_id": cfg["study"]["id"],
        "phase": "phase1",
        "n_pairs": int(len(pair_features)),
        "n_unique_compounds": int(pair_features["compound_id"].nunique()),
        "n_targets": int(pair_features["task"].nunique()),
        "compound_feature_dim": int(len(cmp_cols)),
        "protein_feature_dim": int(len(prot_cols)),
        "tasks": {},
    }
    for task, task_df in pair_features.groupby("task"):
        summary["tasks"][task] = {
            "pairs": int(len(task_df)),
            "positives": int(task_df["label"].sum()),
            "negatives": int((1 - task_df["label"]).sum()),
            "unique_compounds": int(task_df["compound_id"].nunique()),
            "unique_scaffolds": int(task_df["murcko_scaffold"].fillna("MISSING").nunique()),
        }

    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Compound-Protein Interaction Phase 1 Dataset",
        "",
        f"- pairs: `{summary['n_pairs']}`",
        f"- unique compounds: `{summary['n_unique_compounds']}`",
        f"- targets: `{summary['n_targets']}`",
        f"- compound feature dim: `{summary['compound_feature_dim']}`",
        f"- protein feature dim: `{summary['protein_feature_dim']}`",
        "",
        "## Task coverage",
        "",
    ]
    for task, payload in summary["tasks"].items():
        lines.extend(
            [
                f"- `{task}`",
                f"  - pairs: `{payload['pairs']}`",
                f"  - positives: `{payload['positives']}`",
                f"  - negatives: `{payload['negatives']}`",
                f"  - unique compounds: `{payload['unique_compounds']}`",
                f"  - unique scaffolds: `{payload['unique_scaffolds']}`",
            ]
        )
    (out_dir / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_dir / 'pair_features.csv'}")
    print(f"wrote {out_dir / 'pairs.csv'}")
    print(f"wrote {out_dir / 'protein_features.csv'}")
    print(f"wrote {out_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()

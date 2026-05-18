from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from build_target_mie_task_layer import classify_chembl_mechanisms, collect_task_columns


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_source_labels(cfg: dict, task_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    molecules = pd.read_csv(cfg["inputs"]["chembl_molecules_all"]).copy()
    activity_df = pd.read_csv(cfg["inputs"]["chembl_activity_rows"]).copy()

    axis_map = {item["chembl_axis"]: item for item in cfg["targets"]}
    activity_df = activity_df.loc[activity_df["axis"].isin(axis_map)].copy()
    activity_df["target_prefix"] = activity_df["axis"].map(lambda value: axis_map[value]["target_prefix"])
    activity_df["label_column"] = activity_df["axis"].map(lambda value: axis_map[value]["base_label_column"])
    activity_df["pchembl_value_num"] = pd.to_numeric(activity_df["pchembl_value_num"], errors="coerce")
    active_threshold = float(cfg["labeling"]["source_active_if_pchembl_gte"])
    activity_df["active_call"] = activity_df["pchembl_value_num"].ge(active_threshold).fillna(False).astype(int)
    activity_df["mechanisms"] = activity_df.apply(classify_chembl_mechanisms, axis=1)

    labels = molecules[
        [
            "molecule_chembl_id",
            "pref_name",
            "canonical_smiles",
            "standard_inchi_key",
            "full_mwt",
            "full_molformula",
            "alogp",
            "np_likeness_score",
            "natural_product",
        ]
    ].copy()
    labels["compound_id"] = labels["molecule_chembl_id"].astype(str)
    labels["name"] = labels["pref_name"]
    labels["domain"] = np.where(labels["natural_product"].fillna(0).eq(1), "natural_product", "generic_bioactive")
    labels["chemical_class"] = pd.NA
    labels["label_source"] = "chembl_source_target_mie_curated"
    labels["smiles_final"] = labels["canonical_smiles"]
    labels["standard_id"] = labels["standard_inchi_key"]

    for task in task_columns:
        labels[task] = pd.NA
        labels[f"{task}_max_pchembl"] = pd.NA

    base_summary = (
        activity_df.groupby(["molecule_chembl_id", "label_column"], dropna=False)
        .agg(
            n_active=("active_call", "sum"),
            max_pchembl=("pchembl_value_num", "max"),
        )
        .reset_index()
    )
    for target in cfg["targets"]:
        task = target["base_label_column"]
        subset = base_summary.loc[base_summary["label_column"].eq(task)].copy()
        if subset.empty:
            continue
        label_map = subset.set_index("molecule_chembl_id")["n_active"].ge(1).astype(int).to_dict()
        potency_map = subset.set_index("molecule_chembl_id")["max_pchembl"].to_dict()
        labels[task] = labels["molecule_chembl_id"].map(label_map)
        labels[f"{task}_max_pchembl"] = labels["molecule_chembl_id"].map(potency_map)

    mechanism_rows = []
    for row in activity_df.itertuples(index=False):
        for mechanism in row.mechanisms:
            mechanism_rows.append(
                {
                    "molecule_chembl_id": row.molecule_chembl_id,
                    "target_prefix": row.target_prefix,
                    "mechanism": mechanism,
                    "pchembl_value_num": row.pchembl_value_num,
                    "active_call": int(pd.notna(row.pchembl_value_num) and row.pchembl_value_num >= active_threshold),
                }
            )
    mechanism_df = pd.DataFrame(mechanism_rows)
    if not mechanism_df.empty:
        mechanism_summary = (
            mechanism_df.groupby(["molecule_chembl_id", "target_prefix", "mechanism"], dropna=False)
            .agg(
                n_active=("active_call", "sum"),
                max_pchembl=("pchembl_value_num", "max"),
            )
            .reset_index()
        )
    else:
        mechanism_summary = pd.DataFrame(columns=["molecule_chembl_id", "target_prefix", "mechanism", "n_active", "max_pchembl"])

    for target in cfg["targets"]:
        prefix = target["target_prefix"]
        for mechanism in target["mechanisms"]:
            task = f"{prefix}_{mechanism}"
            subset = mechanism_summary.loc[
                mechanism_summary["target_prefix"].eq(prefix) & mechanism_summary["mechanism"].eq(mechanism)
            ].copy()
            if subset.empty:
                continue
            label_map = subset.set_index("molecule_chembl_id")["n_active"].ge(1).astype(int).to_dict()
            potency_map = subset.set_index("molecule_chembl_id")["max_pchembl"].to_dict()
            labels[task] = labels["molecule_chembl_id"].map(label_map)
            labels[f"{task}_max_pchembl"] = labels["molecule_chembl_id"].map(potency_map)

    for mechanism, global_task in cfg["global_mechanism_tasks"].items():
        mechanism_tasks = [f"{target['target_prefix']}_{mechanism}" for target in cfg["targets"] if mechanism in target["mechanisms"]]
        measured = labels[mechanism_tasks].notna().any(axis=1)
        positive = labels[mechanism_tasks].fillna(0).max(axis=1).astype(int)
        labels[global_task] = np.where(measured, positive, pd.NA)
        labels[f"{global_task}_max_pchembl"] = labels[[f"{task}_max_pchembl" for task in mechanism_tasks]].max(axis=1, skipna=True)

    labels["measured_tasks"] = labels[task_columns].apply(
        lambda row: ",".join(task for task in task_columns if pd.notna(row[task])),
        axis=1,
    )

    summary = {
        "n_compounds": int(len(labels)),
        "n_natural_products": int((labels["domain"] == "natural_product").sum()),
        "n_generic_bioactives": int((labels["domain"] == "generic_bioactive").sum()),
        "task_counts": {
            task: {
                "labeled": int(labels[task].notna().sum()),
                "positive": int(pd.to_numeric(labels[task], errors="coerce").fillna(0).sum()),
                "np_labeled": int(((labels["domain"] == "natural_product") & labels[task].notna()).sum()),
                "generic_labeled": int(((labels["domain"] == "generic_bioactive") & labels[task].notna()).sum()),
            }
            for task in task_columns
        },
    }
    return labels, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed ChEMBL source target/MIE labels for natural products and generic bioactives.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    output_dir = Path(cfg["outputs"]["source_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    task_columns = collect_task_columns(cfg["targets"], cfg["global_mechanism_tasks"])
    manifest = {
        "all_tasks": task_columns,
        "targets": cfg["targets"],
        "global_mechanism_tasks": cfg["global_mechanism_tasks"],
    }

    labels, summary = build_source_labels(cfg, task_columns)
    labels.to_csv(output_dir / "chembl_source_labels.csv", index=False)
    (output_dir / "task_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def classify_ctx_mechanism(row: pd.Series) -> str | None:
    function_type = normalize_text(row.get("assay_function_type"))
    endpoint_text = " ".join(
        normalize_text(row.get(column))
        for column in (
            "assay_component_endpoint_name",
            "assay_component_name",
            "assay_component_target_desc",
            "assay_component_endpoint_desc",
        )
    )

    if function_type == "binding":
        return "binding"
    if function_type == "agonist":
        return "agonism"
    if function_type == "antagonist":
        return "antagonism"
    if function_type == "reporter gene":
        return "reporter"
    if function_type == "ratio":
        if "antagon" in endpoint_text:
            return "antagonism"
        if "agon" in endpoint_text:
            return "agonism"
        return "reporter"
    if "binding" in endpoint_text:
        return "binding"
    if "antagon" in endpoint_text:
        return "antagonism"
    if "agon" in endpoint_text:
        return "agonism"
    if any(token in endpoint_text for token in ("reporter", "response element", "transcription", "transactivation", "luc")):
        return "reporter"
    return None


def classify_chembl_mechanisms(row: pd.Series) -> list[str]:
    assay_type = normalize_text(row.get("assay_type")).upper()
    text = " ".join(
        normalize_text(row.get(column))
        for column in ("assay_description", "activity_comment", "action_type")
    )

    mechanisms: set[str] = set()
    if assay_type == "B" or "binding" in text or "ligand binding" in text:
        mechanisms.add("binding")
    if (
        "antagon" in text
        or "inhibition of transcriptional activation" in text
        or "in presence of" in text
        or "against 10 pm" in text
        or "against 0.1 nm" in text
        or "against 0.5 nm" in text
    ):
        mechanisms.add("antagonism")
    if "agon" in text or "activation of" in text or "transcriptional activation" in text:
        mechanisms.add("agonism")
    if assay_type == "F" or any(token in text for token in ("reporter", "luciferase", "response element", "transcriptional", "transactivation")):
        mechanisms.add("reporter")
    return sorted(mechanisms)


def collect_task_columns(target_cfgs: list[dict], global_map: dict[str, str]) -> list[str]:
    tasks: list[str] = []
    for target in target_cfgs:
        tasks.append(target["base_label_column"])
    for target in target_cfgs:
        prefix = target["target_prefix"]
        for mechanism in target["mechanisms"]:
            tasks.append(f"{prefix}_{mechanism}")
    tasks.extend(global_map.values())
    return tasks


def build_pollutant_labels(cfg: dict, task_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    endpoint_df = pd.read_csv(cfg["inputs"]["pollutant_endpoint_catalog"]).copy()
    if "selected" in endpoint_df.columns:
        endpoint_df = endpoint_df.loc[endpoint_df["selected"].fillna(False).astype(bool)].copy()
    bioactivity_df = pd.read_csv(cfg["inputs"]["pollutant_bioactivity_rows"]).copy()
    detail_df = pd.read_csv(cfg["inputs"]["pollutant_chemical_details"]).copy()

    axis_map = {item["axis"]: item for item in cfg["targets"]}
    endpoint_df["target_prefix"] = endpoint_df["axis"].map(lambda value: axis_map[value]["target_prefix"])
    endpoint_df["mechanism"] = endpoint_df.apply(classify_ctx_mechanism, axis=1)

    active_threshold = float(cfg["labeling"]["pollutant_active_if_hitc_gte"])
    bioactivity_df["hitc_num"] = pd.to_numeric(bioactivity_df["hitc"], errors="coerce")
    bioactivity_df["active_call"] = bioactivity_df["hitc_num"].fillna(0) >= active_threshold
    bioactivity_df = bioactivity_df.merge(
        endpoint_df[
            [
                "aeid",
                "axis",
                "label_column",
                "target_prefix",
                "mechanism",
                "assay_function_type",
                "official_symbol",
            ]
        ],
        on=["aeid", "axis", "label_column"],
        how="left",
    )

    labels = detail_df[
        [
            "dtxsid",
            "preferredName",
            "casrn",
            "dtxcid",
            "molFormula",
            "smiles",
            "qsarReadySmiles",
            "msReadySmiles",
            "qcLevel",
            "qcLevelDesc",
            "iupacName",
            "activeAssays",
            "totalAssays",
            "percentAssays",
        ]
    ].copy()
    labels = labels.rename(
        columns={
            "preferredName": "name",
            "molFormula": "mol_formula",
            "smiles": "smiles_ctx",
            "qsarReadySmiles": "qsar_ready_smiles",
            "msReadySmiles": "ms_ready_smiles",
            "iupacName": "iupac_name",
            "activeAssays": "active_assays_ctx",
            "totalAssays": "total_assays_ctx",
            "percentAssays": "percent_assays_ctx",
        }
    )
    labels["smiles_final"] = (
        labels["qsar_ready_smiles"].fillna(labels["smiles_ctx"]).fillna(labels["ms_ready_smiles"])
    )
    labels["domain"] = "pollutant"
    labels["compound_id"] = labels["dtxsid"].astype(str)
    labels["chemical_class"] = pd.NA
    labels["chembl_id"] = pd.NA
    labels["standard_id"] = pd.NA
    labels["label_source"] = "ctx_target_mie_curated_endpoints"

    for task in task_columns:
        labels[task] = pd.NA

    target_axis_summary = (
        bioactivity_df.groupby(["dtxsid", "label_column"], dropna=False)
        .agg(n_active=("active_call", "sum"), n_rows=("aeid", "size"))
        .reset_index()
    )
    for target in cfg["targets"]:
        task = target["base_label_column"]
        subset = target_axis_summary.loc[target_axis_summary["label_column"].eq(task)].copy()
        if subset.empty:
            continue
        task_map = subset.set_index("dtxsid")["n_active"].ge(1).astype(int).to_dict()
        labels[task] = labels["dtxsid"].map(task_map)

    mechanism_summary = (
        bioactivity_df.loc[bioactivity_df["mechanism"].notna()]
        .groupby(["dtxsid", "target_prefix", "mechanism"], dropna=False)
        .agg(n_active=("active_call", "sum"), n_rows=("aeid", "size"), max_hitc=("hitc_num", "max"))
        .reset_index()
    )
    for target in cfg["targets"]:
        prefix = target["target_prefix"]
        for mechanism in target["mechanisms"]:
            task = f"{prefix}_{mechanism}"
            subset = mechanism_summary.loc[
                mechanism_summary["target_prefix"].eq(prefix) & mechanism_summary["mechanism"].eq(mechanism)
            ].copy()
            if subset.empty:
                continue
            task_map = subset.set_index("dtxsid")["n_active"].ge(1).astype(int).to_dict()
            labels[task] = labels["dtxsid"].map(task_map)

    for mechanism, global_task in cfg["global_mechanism_tasks"].items():
        mechanism_tasks = [f"{target['target_prefix']}_{mechanism}" for target in cfg["targets"] if mechanism in target["mechanisms"]]
        measured = labels[mechanism_tasks].notna().any(axis=1)
        positive = labels[mechanism_tasks].fillna(0).max(axis=1).astype(int)
        labels[global_task] = np.where(measured, positive, pd.NA)

    labels["measured_tasks"] = labels[task_columns].apply(
        lambda row: ",".join(task for task in task_columns if pd.notna(row[task])),
        axis=1,
    )

    summary = {
        "n_compounds": int(len(labels)),
        "task_counts": {
            task: {
                "labeled": int(labels[task].notna().sum()),
                "positive": int(pd.to_numeric(labels[task], errors="coerce").fillna(0).sum()),
            }
            for task in task_columns
        },
        "mechanism_endpoint_counts": (
            endpoint_df.groupby(["target_prefix", "mechanism"]).size().reset_index(name="n_endpoints").to_dict("records")
        ),
    }
    return labels, summary


def build_np_labels(cfg: dict, task_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    base_labels = pd.read_csv(cfg["inputs"]["natural_product_base_labels"]).copy()
    activity_df = pd.read_csv(cfg["inputs"]["natural_product_activity_rows"]).copy()

    chembl_axis_map = {item["chembl_axis"]: item for item in cfg["targets"]}
    activity_df["target_prefix"] = activity_df["axis"].map(lambda value: chembl_axis_map[value]["target_prefix"])
    activity_df["pchembl_value_num"] = pd.to_numeric(activity_df["pchembl_value_num"], errors="coerce")
    active_threshold = float(cfg["labeling"]["natural_product_active_if_pchembl_gte"])
    activity_df["mechanisms"] = activity_df.apply(classify_chembl_mechanisms, axis=1)

    labels = base_labels.copy()
    for task in task_columns:
        if task not in labels.columns:
            labels[task] = pd.NA
        potency_col = f"{task}_max_pchembl"
        if potency_col not in labels.columns:
            labels[potency_col] = pd.NA

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
                n_rows=("mechanism", "size"),
                n_active=("active_call", "sum"),
                max_pchembl=("pchembl_value_num", "max"),
            )
            .reset_index()
        )
    else:
        mechanism_summary = pd.DataFrame(columns=["molecule_chembl_id", "target_prefix", "mechanism", "n_rows", "n_active", "max_pchembl"])

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
        global_potency_col = f"{global_task}_max_pchembl"
        labels[global_potency_col] = labels[[f"{task}_max_pchembl" for task in mechanism_tasks]].max(axis=1, skipna=True)

    labels["measured_tasks"] = labels[task_columns].apply(
        lambda row: ",".join(task for task in task_columns if pd.notna(row[task])),
        axis=1,
    )
    labels["label_source"] = "chembl_multispecies_target_mie_curated"

    summary = {
        "n_compounds": int(len(labels)),
        "task_counts": {
            task: {
                "labeled": int(labels[task].notna().sum()),
                "positive": int(pd.to_numeric(labels[task], errors="coerce").fillna(0).sum()),
            }
            for task in task_columns
        },
        "mechanism_activity_rows": int(len(mechanism_df)),
    }
    return labels, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a target/MIE mechanism task layer from CTX and ChEMBL data.")
    parser.add_argument("--config", required=True, help="Path to target/MIE task-layer YAML config.")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    pollutant_dir = Path(cfg["outputs"]["pollutant_dir"])
    natural_product_dir = Path(cfg["outputs"]["natural_product_dir"])
    pollutant_dir.mkdir(parents=True, exist_ok=True)
    natural_product_dir.mkdir(parents=True, exist_ok=True)

    task_columns = collect_task_columns(cfg["targets"], cfg["global_mechanism_tasks"])
    manifest = {
        "base_target_tasks": [item["base_label_column"] for item in cfg["targets"]],
        "target_mechanism_tasks": {
            item["target_prefix"]: [f"{item['target_prefix']}_{mechanism}" for mechanism in item["mechanisms"]]
            for item in cfg["targets"]
        },
        "global_mechanism_tasks": cfg["global_mechanism_tasks"],
        "all_tasks": task_columns,
    }

    pollutant_labels, pollutant_summary = build_pollutant_labels(cfg, task_columns)
    pollutant_labels.to_csv(pollutant_dir / "ctx_labels.csv", index=False)
    (pollutant_dir / "task_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (pollutant_dir / "summary.json").write_text(json.dumps(pollutant_summary, indent=2) + "\n", encoding="utf-8")

    np_labels, np_summary = build_np_labels(cfg, task_columns)
    np_labels.to_csv(natural_product_dir / "chembl_np_labels.csv", index=False)
    (natural_product_dir / "task_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (natural_product_dir / "summary.json").write_text(json.dumps(np_summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

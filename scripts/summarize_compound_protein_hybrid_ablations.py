#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


RUNS = {
    "rf_reference": ("outputs/compound_protein_interaction_retrievalhybrid_v1_full5/aggregate_metrics.csv", "rf_reference_single_task"),
    "seq_only": ("outputs/compound_protein_interaction_crossattn_v1_full5/aggregate_metrics.csv", "cross_attention_cpi"),
    "single_pollutant": ("outputs/compound_protein_interaction_retrievalhybrid_pollutant_v1/aggregate_metrics.csv", "retrieval_augmented_seq_cpi"),
    "single_generic": ("outputs/compound_protein_interaction_retrievalhybrid_generic_v1/aggregate_metrics.csv", "retrieval_augmented_seq_cpi"),
    "single_mixed": ("outputs/compound_protein_interaction_retrievalhybrid_mixed_v1/aggregate_metrics.csv", "retrieval_augmented_seq_cpi"),
    "multi_memory": ("outputs/compound_protein_interaction_retrievalhybrid_v1_full5/aggregate_metrics.csv", "retrieval_augmented_seq_cpi"),
}


def main() -> None:
    out_dir = Path("outputs/compound_protein_interaction_hybrid_ablations")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for label, (path_str, model_name) in RUNS.items():
        df = pd.read_csv(path_str)
        df = df[df["model"] == model_name].copy()
        df["run_label"] = label
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "ablation_metrics.csv", index=False)

    rf = combined[combined["run_label"] == "rf_reference"][["task", "pr_auc_mean", "precision_at_5pct_mean", "enrichment_factor_5pct_mean"]].rename(
        columns={
            "pr_auc_mean": "rf_pr_auc",
            "precision_at_5pct_mean": "rf_p5",
            "enrichment_factor_5pct_mean": "rf_ef5",
        }
    )
    summary = combined.merge(rf, on="task", how="left")
    summary["delta_pr_auc_vs_rf"] = summary["pr_auc_mean"] - summary["rf_pr_auc"]
    summary["delta_p5_vs_rf"] = summary["precision_at_5pct_mean"] - summary["rf_p5"]
    summary["delta_ef5_vs_rf"] = summary["enrichment_factor_5pct_mean"] - summary["rf_ef5"]
    summary = summary.sort_values(["task", "run_label"]).reset_index(drop=True)
    summary.to_csv(out_dir / "ablation_summary_vs_rf.csv", index=False)

    lines = ["# Hybrid Ablation Summary", ""]
    for task, task_df in summary.groupby("task"):
        lines.append(f"## {task}")
        best = task_df.sort_values("pr_auc_mean", ascending=False).iloc[0]
        lines.append(f"- best run: `{best['run_label']}` (PR-AUC `{best['pr_auc_mean']:.4f}`)")
        for _, row in task_df.iterrows():
            lines.append(
                f"- `{row['run_label']}`: PR-AUC `{row['pr_auc_mean']:.4f}` "
                f"(delta vs RF `{row['delta_pr_auc_vs_rf']:+.4f}`), "
                f"P@5% `{row['precision_at_5pct_mean']:.4f}`, EF@5% `{row['enrichment_factor_5pct_mean']:.4f}`"
            )
        lines.append("")
    (out_dir / "status.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

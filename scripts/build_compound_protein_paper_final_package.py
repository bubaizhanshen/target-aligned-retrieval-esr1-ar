from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "compound_protein_interaction_paper_final_package"


def build_main_method_table() -> pd.DataFrame:
    env = pd.read_csv(
        ROOT / "outputs" / "external_method_comparison_v2" / "environmental_method_comparison.csv"
    )
    best = pd.read_csv(
        ROOT / "outputs" / "compound_protein_interaction_hybrid_v2_final" / "best_overall_summary.csv"
    )
    best = best[["task", "setting", "relative_improvement_pct", "hybrid_p5", "rf_p5"]]

    wide = env.pivot(index="task", columns="model", values="pr_auc_mean").reset_index()
    wide.columns.name = None
    out = wide.merge(best, on="task", how="left")
    rename = {
        "hybrid": "hybrid_pr_auc",
        "rf": "rf_pr_auc",
        "drugban": "drugban_pr_auc",
        "moltrans": "moltrans_pr_auc",
    }
    out = out.rename(columns=rename)
    cols = [
        "task",
        "setting",
        "hybrid_pr_auc",
        "rf_pr_auc",
        "drugban_pr_auc",
        "moltrans_pr_auc",
        "relative_improvement_pct",
        "hybrid_p5",
        "rf_p5",
    ]
    return out[cols].sort_values("task").reset_index(drop=True)


def build_ablation_table() -> pd.DataFrame:
    abl = pd.read_csv(
        ROOT
        / "outputs"
        / "compound_protein_interaction_hybrid_ablations"
        / "ablation_summary_vs_rf.csv"
    )
    keep = [
        "multi_memory",
        "single_mixed",
        "single_pollutant",
        "single_generic",
        "seq_only",
        "rf_reference",
    ]
    out = abl[abl["run_label"].isin(keep)].copy()
    cols = [
        "task",
        "run_label",
        "pr_auc_mean",
        "precision_at_5pct_mean",
        "enrichment_factor_5pct_mean",
        "delta_pr_auc_vs_rf",
        "delta_p5_vs_rf",
        "delta_ef5_vs_rf",
    ]
    return out[cols].sort_values(["task", "pr_auc_mean"], ascending=[True, False]).reset_index(drop=True)


def build_case_study_table() -> pd.DataFrame:
    rescued_ar = pd.read_csv(
        ROOT / "outputs" / "compound_protein_interaction_hybrid_case_analysis" / "ar_binding_rescued.csv"
    )
    rescued_esr1 = pd.read_csv(
        ROOT / "outputs" / "compound_protein_interaction_hybrid_case_analysis" / "esr1_binding_rescued.csv"
    )
    support = pd.read_csv(
        ROOT
        / "outputs"
        / "compound_protein_interaction_hybrid_paper_ready"
        / "retrieval_support_summary_all.csv"
    )

    selected = []
    for task, df, k in [
        ("ar_binding", rescued_ar, 4),
        ("esr1_binding", rescued_esr1, 4),
    ]:
        chosen = (
            df.sort_values("delta_score", ascending=False)
            .drop_duplicates("compound_id")
            .head(k)
            .copy()
        )
        chosen["task"] = task
        selected.append(chosen)

    cases = pd.concat(selected, ignore_index=True)
    support_task = support.copy()

    piv = support_task.pivot_table(
        index=["task", "query_compound_id"],
        columns="memory_name",
        values=["top_ref_name", "top_ref_similarity", "mean_ref_label"],
        aggfunc="first",
    )
    piv.columns = [f"{metric}_{memory}" for metric, memory in piv.columns.to_flat_index()]
    piv = piv.reset_index()

    merged = cases.merge(
        piv,
        left_on=["task", "compound_id"],
        right_on=["task", "query_compound_id"],
        how="left",
    )

    keep_cols = [
        "task",
        "compound_id",
        "name",
        "delta_score",
        "hybrid_score",
        "rf_score",
        "murcko_scaffold",
        "top_ref_name_pollutant_only",
        "top_ref_similarity_pollutant_only",
        "mean_ref_label_pollutant_only",
        "top_ref_name_generic_bioactive",
        "top_ref_similarity_generic_bioactive",
        "mean_ref_label_generic_bioactive",
        "top_ref_name_mixed",
        "top_ref_similarity_mixed",
        "mean_ref_label_mixed",
        "top_ref_name_ar_support_v2",
        "top_ref_similarity_ar_support_v2",
        "mean_ref_label_ar_support_v2",
    ]
    available = [c for c in keep_cols if c in merged.columns]
    return merged[available].sort_values(["task", "delta_score"], ascending=[True, False]).reset_index(drop=True)


def build_application_shortlist() -> pd.DataFrame:
    shortlist = pd.read_csv(
        ROOT / "outputs" / "compound_target_final_package" / "external_priority_shortlist.csv"
    )
    review = pd.read_csv(
        ROOT / "outputs" / "compound_target_manual_review_v1" / "manual_review_master.csv"
    )

    review_cols = [
        "dtxsid",
        "review_bucket",
        "followup_action",
        "review_priority",
        "panel_group",
        "followup_target",
        "in_validation_panel",
    ]
    merged = shortlist.merge(review[review_cols], on="dtxsid", how="left")
    return merged.sort_values(
        ["application_track", "max_target_score"], ascending=[True, False]
    ).reset_index(drop=True)


def build_public_benchmark_note() -> pd.DataFrame:
    davis = pd.read_csv(
        ROOT / "outputs" / "external_method_comparison_v2" / "davis_method_comparison.csv"
    )
    return davis.sort_values(["split", "rmse"]).reset_index(drop=True)


def write_status(
    main_table: pd.DataFrame,
    ablation_table: pd.DataFrame,
    case_table: pd.DataFrame,
    app_table: pd.DataFrame,
    davis_table: pd.DataFrame,
) -> None:
    lines = [
        "# Compound-Protein Interaction Paper Final Package",
        "",
        "## Main Result",
        "",
    ]
    for _, row in main_table.sort_values("task").iterrows():
        lines.append(
            f"- `{row['task']}`: hybrid `{row['hybrid_pr_auc']:.4f}` vs RF `{row['rf_pr_auc']:.4f}` vs DrugBAN `{row['drugban_pr_auc']:.4f}` vs MolTrans `{row['moltrans_pr_auc']:.4f}`; relative improvement over RF `+{row['relative_improvement_pct']:.2f}%`."
        )

    lines.extend(
        [
            "",
            "## Ablation",
            "",
            "- `multi_memory` is the best retrieval setting for both core tasks.",
            "- `seq_only` is consistently weaker, so retrieval evidence is necessary rather than optional.",
            "",
            "## Case Study Package",
            "",
            f"- Representative rescued compounds included: `{', '.join(case_table['name'].head(6).tolist())}`.",
            "- Case tables include hybrid-vs-RF score lift and top retrieval support from each memory.",
            "",
            "## Environmental Application Package",
            "",
            f"- External shortlist contains `{len(app_table)}` candidates with mechanism route, review bucket, and follow-up recommendation.",
            f"- Validation-panel overlap: `{int(app_table['in_validation_panel'].fillna(False).sum())}` candidates.",
            "",
            "## Public Benchmark Note",
            "",
        ]
    )
    for split in davis_table["split"].drop_duplicates():
        sub = davis_table[davis_table["split"] == split]
        best = sub.iloc[0]
        lines.append(
            f"- `DAVIS {split}`: best RMSE is `{best['method']}` at `{best['rmse']:.4f}`."
        )

    (OUT_DIR / "status.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_table = build_main_method_table()
    ablation_table = build_ablation_table()
    case_table = build_case_study_table()
    app_table = build_application_shortlist()
    davis_table = build_public_benchmark_note()

    main_table.to_csv(OUT_DIR / "main_method_table.csv", index=False)
    ablation_table.to_csv(OUT_DIR / "ablation_table.csv", index=False)
    case_table.to_csv(OUT_DIR / "case_study_table.csv", index=False)
    app_table.to_csv(OUT_DIR / "application_shortlist_table.csv", index=False)
    davis_table.to_csv(OUT_DIR / "public_benchmark_note.csv", index=False)
    write_status(main_table, ablation_table, case_table, app_table, davis_table)


if __name__ == "__main__":
    main()

# Target-Aligned Retrieval for ESR1/AR Environmental Binding Prioritization

This repository contains the code, processed datasets, and result tables used for the ESR1/AR direct-binding prioritization study on environmental suspect chemicals.

It intentionally excludes manuscript files, figures, plotting scripts, and raw bulky source tables. The goal here is to provide a clean code-and-data release that supports method inspection and result reproduction without carrying the full drafting workspace.

## Scope

- Primary targets: `ESR1` and `AR`
- Primary benchmark endpoint: direct-binding prioritization
- Benchmark construction:
  - pollutant-side receptor labels retained at `CTX hitc >= 0.9`
  - external source-domain reference labels retained at `ChEMBL pChEMBL >= 5.0`
  - final benchmark restricted to `esr1_binding` and `ar_binding`
- Split protocol:
  - five scaffold-holdout resamples
  - group-aware repeated scaffold resampling rather than a strict disjoint 5-fold CV

## Repository Layout

- `configs/`
  Model and dataset configuration files for benchmark assembly and model runs.

- `scripts/`
  Data-building, model-training, baseline, and audit scripts.

- `data/`
  Processed benchmark labels, compound/target panels, pair-level features, and retrieval banks.

- `outputs/`
  Paper-facing result tables, robustness audits, external comparisons, and the frozen binding-first panel tables.

- `requirements.txt`
  Minimal Python package requirements tracked in the working environment.

- `LICENSE` and `CITATION.cff`
  Reuse and citation metadata for the public code-and-data release.

## Included Processed Data

- `data/ctx_target_mie_er_ahr_ar/`
  Curated pollutant-side receptor label layer.

- `data/chembl_source_target_mie_er_ahr_ar/`
  Curated external source-domain label layer.

- `data/compound_target_phase2_v2/`
  Standardized compound set used for the retained target tasks.

- `data/compound_protein_interaction_v1/`
  Retained target panel metadata.

- `data/compound_protein_interaction_v1_phase1/`
  Pair table and pair/protein features used for benchmarking.

- `data/compound_target_retrieval_v1_bank/`
  Primary retrieval bank for ESR1/AR target-aligned retrieval.

- `data/compound_target_retrieval_ar_v2_bank/`
  Refined AR-aligned retrieval bank used in AR support analyses.

## Included Result Tables

- `outputs/compound_protein_interaction_paper_final_package/`
  Main benchmark, ablation, case-study, and shortlist tables.

- `outputs/compound_protein_interaction_hybrid_paper_ready/`
  Paired predictions, resample summaries, chemotype summaries, and retrieval-support summaries.

- `outputs/compound_protein_interaction_review_defenses/`
  Leakage, calibration, applicability, and read-across audit tables.

- `outputs/external_method_comparison_v2/`
  Environmental and DAVIS external comparison summaries.

- `outputs/compound_protein_interaction_binding_first_panel_v1/`
  Frozen binding-first panel tables, including the authoritative 15-compound panel.

## What Is Not Included

- Manuscript and supporting-information documents
- Figures and figure-generation scripts
- Raw high-volume source tables used only during exploratory curation
- Large intermediate drafting assets

## High-Level Reproduction Order

1. Build the pollutant-side receptor task layer.
2. Build the external source-domain label layer.
3. Assemble the retained compound/target panel and pair-level benchmark.
4. Run baseline and retrieval-hybrid models.
5. Run robustness and review-defense audits.
6. Build the final paper-facing result tables.

Representative entry points:

- `scripts/build_target_mie_task_layer.py`
- `scripts/build_chembl_source_target_mie_labels.py`
- `scripts/build_compound_protein_target_panel.py`
- `scripts/build_compound_protein_interaction_phase1_dataset.py`
- `scripts/run_compound_protein_interaction_baselines.py`
- `scripts/run_compound_protein_interaction_retrievalhybrid.py`
- `scripts/analyze_compound_protein_review_defenses.py`
- `scripts/build_compound_protein_paper_final_package.py`

## Notes

- This repository is a curated release subset, not the entire working directory.
- Some scripts assume the relative directory structure shipped here.
- The frozen binding-first panel should be read from:
  - `outputs/compound_protein_interaction_binding_first_panel_v1/frozen_binding_first_panel.csv`
  rather than inferred from broader shortlist tables.

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd


TARGETS = [
    {
        "task": "esr1_binding",
        "gene_symbol": "ESR1",
        "uniprot_accession": "P03372",
        "target_family": "nuclear_receptor",
        "role": "core",
        "environmental_relevance": "endocrine_disruption_estrogenic",
    },
    {
        "task": "ar_binding",
        "gene_symbol": "AR",
        "uniprot_accession": "P10275",
        "target_family": "nuclear_receptor",
        "role": "core",
        "environmental_relevance": "endocrine_disruption_androgenic",
    },
    {
        "task": "ahr_binding",
        "gene_symbol": "AHR",
        "uniprot_accession": "P35869",
        "target_family": "bhlh_pas_receptor",
        "role": "expansion",
        "environmental_relevance": "xenobiotic_response_ahr",
    },
    {
        "task": "pparg_binding",
        "gene_symbol": "PPARG",
        "uniprot_accession": "P37231",
        "target_family": "nuclear_receptor",
        "role": "expansion",
        "environmental_relevance": "metabolic_disruption_pparg",
    },
]


def fetch_uniprot_record(accession: str) -> dict:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main() -> None:
    out_dir = Path("data/compound_protein_interaction_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in TARGETS:
        record = fetch_uniprot_record(target["uniprot_accession"])
        protein_name = record["proteinDescription"]["recommendedName"]["fullName"]["value"]
        sequence = record["sequence"]["value"]
        organism = record["organism"]["scientificName"]
        rows.append(
            {
                **target,
                "protein_name": protein_name,
                "organism": organism,
                "sequence_length": len(sequence),
                "protein_sequence": sequence,
                "uniprot_url": f"https://www.uniprot.org/uniprotkb/{target['uniprot_accession']}",
            }
        )

    panel = pd.DataFrame(rows)
    panel.to_csv(out_dir / "target_panel.csv", index=False)

    summary = {
        "study_id": "compound_protein_interaction_v1",
        "n_targets": int(len(panel)),
        "core_targets": panel.loc[panel["role"] == "core", "gene_symbol"].tolist(),
        "expansion_targets": panel.loc[panel["role"] == "expansion", "gene_symbol"].tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Compound-Protein Interaction Target Panel",
        "",
        f"- total targets: `{len(panel)}`",
        f"- core targets: `{', '.join(summary['core_targets'])}`",
        f"- expansion targets: `{', '.join(summary['expansion_targets'])}`",
        "",
        "## Included Targets",
        "",
    ]
    for _, row in panel.iterrows():
        lines.append(
            f"- `{row['gene_symbol']}` / `{row['uniprot_accession']}` / "
            f"`{row['protein_name']}` / `{row['sequence_length']}` aa / `{row['role']}`"
        )
    (out_dir / "target_panel.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

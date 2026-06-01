# Contributing

This repository is a curated code-and-data release for the ESR1/AR
environmental binding prioritization workflow. Contributions are welcome when
they improve reproducibility, documentation, data provenance, or benchmark
auditing.

## Useful Contributions

- Reproduction reports for the scripts listed in `README.md`
- Documentation fixes for setup, data scope, or result-table interpretation
- Bug reports with the command, Python version, package versions, and traceback
- Small tests or validation checks for table schemas and expected output files
- Suggestions for clearer citation or upstream data-source attribution

## Before Opening a Pull Request

1. Keep changes scoped to a single reproducibility or maintenance concern.
2. Do not add manuscript drafts, unpublished private data, credentials, or raw
   bulky source tables.
3. Run the smallest relevant script or validation command you can, and describe
   the result in the pull request.
4. Update `README.md` or `CITATION.cff` when the change affects reuse,
   citation, or reproduction instructions.

## Maintainer Review

Pull requests are reviewed for reproducibility, provenance, and clarity before
merge. For data changes, include enough detail for another maintainer to trace
the source and transformation path.

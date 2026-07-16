# Result artifacts

This directory contains machine-readable evaluations, benchmarks, mechanistic analyses,
and reports derived from run logs. Large checkpoints and raw token files are intentionally
excluded from version control.

Training writes checkpoints and the complete metric stream under
`runs/<suite>/<plan>/`, for example `runs/12l_1k_100m/9a3t/` or
`runs/20l_4k_1b/15a5t/`.
Standalone evaluation and benchmark commands write reviewable JSON under
`results/generated/` when passed `--output`. The primary in-domain result set uses FineWeb-Edu
validation; Wikipedia and other out-of-domain evaluations are optional additions and
should be labeled separately.

Mechanistic analysis commands also write their complete layerwise spectra, residual-update
statistics, attention-head diversity, and intervention results under `results/generated/`.
Keep these raw JSON files alongside any derived rank or sensitivity figures.

Generate derived suite reports with `python -m triglu.report --suite <suite>`. The tool
writes tables and dependency-free SVG charts under `results/generated/<suite>-report/`.
Those files are reproducible presentation artifacts; the run logs, resolved configs, and
raw evaluation/benchmark/analysis JSON remain the scientific record.

The context-scaling sweep writes one labeled benchmark JSON per architecture and context.
The generated report collects these into `context_benchmarks.csv` and editable SVGs while
keeping measurements beyond the trained context explicitly separate from quality claims.

Only artifacts with the provenance required by
[`docs/results-schema.md`](../docs/results-schema.md) belong in a reported comparison.

Original result figures and tables are available under CC BY 4.0; analysis software is
available under Apache-2.0. See the repository's
[licensing scope](../LICENSES/README.md) for details.

# Analysis scripts

These are the scripts that produced the template, the statistics and the figures in
[`docs/report.md`](../docs/report.md). They are kept for provenance, as they were run. They are
not part of the installed package and need extra software: `scipy`, `matplotlib`, `pyvo` and the
`aws` command line client.

They expect to be run from an analysis directory that contains the package checkout as
`spherex-fpwcs/` and writes to `out/`, `cat/` and `data/` next to it, with the scripts in
`scripts/`. Adjust the paths at the top of each file if your layout differs.

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `scan_headers.py` | image headers of every file of one week, by ranged requests | bucket listing | `out/week_headers.jsonl` |
| `longterm_fetch.py` | headers of 120 complete exposures from every 4th half-week of DR1 | bucket | `out/long_headers.jsonl` |
| `build_template.py` | six-detector template and the aberration test | `out/long_headers.jsonl` | `out/fp_template_v1.json` |
| `focalplane.py` | rigidity test within a week; common distortion per focal plane | `out/week_headers.jsonl` | console |
| `longterm_analyze.py` | drift of distortion and geometry over 29 epochs | `out/long_headers.jsonl` | `out/long_template_drift.npy` |
| `pkg_scan.py` | package attitude fit over the cached headers | both header files | `out/pkg_week.json`, `out/pkg_long.json` |
| `get_2mass.py` | 2MASS stars for pointing `2026W13_2A_0568` | IRSA TAP | `cat/` |
| `measure_resid.py` | star offsets per cell under the header WCS | images, `cat/` | `out/resid_*.json` |
| `remeasure_with_standard_sip.py` | detection and cell-measurement helpers; refit with a standard SIP | images, `cat/` | console |
| `pkg_validate.py` | end-to-end star validation of the package | images (downloaded on demand), IRSA TAP | `out/pkg_validate_*.json` |
| `make_figures.py` | figures of the report | `out/` | `docs/figures/` |

The header-only steps need no image downloads. The star validation downloads about 60 MB per
file.

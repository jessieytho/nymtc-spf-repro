# Technical Documentation — `nymtc-spf-repro`

Reproducibility artifact for *From Counts to Rates: A Reproducible Exposure-Normalized
Safety-Performance Method for Small-Area Crash Analysis.*

This document is the technical reference. For a 60-second orientation see `README.md`;
for step-by-step GitHub publication see `PUBLISHING_TO_GITHUB.md`.

---

## 1. What this artifact computes

Given a small-area (county) × year panel of crash, fatality, and serious-injury **counts**
plus matching **exposure** (vehicle-miles traveled, VMT), the pipeline:

1. estimates exposure-normalized **year rate ratios** per outcome with a safety-performance-function
   (SPF) offset GLM and few-cluster **wild cluster bootstrap** inference;
2. demonstrates that a **count-on-count** fixed-effects specification on the same data
   manufactures a spurious, *unidentified* "lethality multiplier" (reproducible ≠ valid);
3. validates both with a known-truth **Monte Carlo** study and an external **national-series** benchmark;
4. **replicates** on an independent region (NJTPA, G = 13) with real few-cluster power;
5. checks **robustness** to exposure-measurement error.

Every table and figure in the paper is regenerated from source by `python run_all.py`.

---

## 2. Method specification

### 2.1 The recommended estimator (SPF-offset GLM)

For county *i* in year *t*, outcome count *y*, exposure VMT, the mean structure is

```
log E[y_it] = log(VMT_it) + alpha_i + gamma_t
```

- the offset coefficient on `log(VMT)` is **fixed at 1** (so the linear predictor models the
  log rate per VMT, not the count);
- `alpha_i` are county fixed effects, `gamma_t` year fixed effects (2019 = reference);
- **estimand:** the year effects. Each `exp(gamma_t)` is an incidence-rate ratio (IRR) vs 2019,
  net of fixed county composition.

Mean parameters are estimated by **Poisson quasi-maximum likelihood (QMLE)** — consistent for the
rate model regardless of the true conditional variance. Overdispersion is *not* modeled by switching
likelihoods; it is carried entirely by the variance estimator (§2.3). A negative-binomial dispersion
and the Pearson dispersion (severe for crashes ≈156, moderate for serious injuries ≈6.7, ≈1.07 for
fatalities) are reported only as diagnostics.

### 2.2 The diagnosed (wrong) estimator (count-on-count)

A severe-outcome count regressed on a less-severe count with county + year fixed effects, the slope
ratio read as a "lethality multiplier." With within (county- and year-demeaned) series, the slope is
`beta_c = sum(F~ C~) / sum(C~^2)` and the multiplier is `M = beta_T / beta_N`. The paper proves
(Proposition 1) that `M` is **unidentified rather than biased**: `beta_c` is unbiased for the true
per-crash rate `theta_c`, but its precision is governed by the within concentration
`SNR_c = n·Var_w(C_c)/sigma_c^2`. When `SNR_N = O(1)` the denominator slope carries sampling mass at/across
zero, so `M` is approximately a ratio of normals — heavy-tailed, no finite variance, frequent sign flips.
Counterintuitively, instability is **worst** in the smooth, high-volume, near-Poisson regime (low
overdispersion) the fatality series occupies. The SPF-offset model cannot produce this artifact because
each outcome is normalized by its own period-matched exposure before any comparison.

### 2.3 Inference (few clusters)

Errors are clustered by county (G = 10 for NYMTC). With few clusters the cluster-robust sandwich with a
normal reference over-rejects, so inference uses a **restricted (null-imposed) wild cluster bootstrap-t**
with Rademacher weights at the county level; each wild sample is refit so coefficient and SE both vary.

- **Intervals invert the test:** the 95% interval for a year effect is the set of null rate ratios the
  bootstrap fails to reject at 5% (grid inversion), so interval and test agree by construction.
- **Exact enumeration:** with G = 10 there are only `2^10 = 1,024` distinct sign vectors; they are
  enumerated exactly → **no simulation error**. Beyond `G ≳ 25` (about 33M vectors) the code falls back
  to `B` seeded sampled Rademacher draws with no change to the calling interface.

### 2.4 Determinism

Exact enumeration carries no randomness; every sampled step (Monte Carlo, sampled bootstrap) is seeded.
Re-runs on identical inputs are bit-for-bit reproducible; a re-run on a new county-year re-runs the full
pipeline (batch, not streaming).

---

## 3. Repository layout

```
nymtc-spf-repro/
├── run_all.py                 # runs src/NN_*.py in numeric order; mirrors outputs/*.png → figures/
├── src/                       # pipeline steps (numeric-ordered)
│   ├── 01_verify_vmt_provenance.py
│   ├── 02_spf_model.py
│   ├── 03_wild_cluster_bootstrap.py
│   ├── 04_grid_inverted_wcr.py
│   ├── 05_fig_divergence.py
│   ├── 06_wrong_vs_right.py
│   ├── 09_simulation_instability.py
│   ├── 10_simulation_coverage.py
│   ├── 11_national_benchmark.py
│   ├── 12_figures_redesign.py
│   ├── 13_worked_example.py
│   ├── 14_second_region.py    # Capital District (G=4) — inferential-floor replication
│   ├── 15_third_region.py     # NJTPA (G=13) — featured replication
│   └── 16_vmt_sensitivity.py
├── tests/                     # no-pytest suite; guards the core claims
│   ├── run_tests.py
│   ├── test_enumeration.py
│   ├── test_provenance.py
│   ├── test_bootstrap_ci.py
│   └── test_simulation_instability.py
├── data/                      # inputs (see §5 and data/README.md)
│   └── README.md              # data provenance: included vs user-supplied, checksums
├── outputs/                   # regenerated CSVs + PNGs (written by run_all.py)
├── figures/                   # PNGs mirrored for the LaTeX build (\graphicspath)
├── Dockerfile                 # pinned env; runs tests as final build step
├── .github/workflows/ci.yml   # tests on py3.11/3.12 + docker build + ruff
├── requirements.txt           # loose lower bounds
├── requirements.lock.txt      # exact pinned versions (CI + Docker)
├── pyproject.toml             # project metadata + ruff config
├── LICENSE                    # code license
├── CITATION.cff               # "Cite this repository"
├── README.md                  # quick start + overview
├── PUBLISHING_TO_GITHUB.md    # first-time GitHub publication walkthrough
└── DOCS.md                    # this file
```

Note `src/` skips 07–08 by design; `run_all.py` globs `src/[0-9][0-9]_*.py` and runs whatever is present,
in order, so gaps are harmless.

---

## 4. Pipeline → paper artifact map

Run order is numeric. "Paper" column gives the table/figure each step backs.

| Step | Script | Key output(s) | Paper |
|---|---|---|---|
| 01 | `01_verify_vmt_provenance.py` | provenance check (VMT ratio 1.0000; live if the NYSDOT DVMT workbook supplied, else documented) | §3.1 claim |
| 02 | `02_spf_model.py` | `nymtc_safety_panel.csv`, `irr_crashes/fatalities/serious.csv`, `exhibit_counts_vs_rates.csv` | Table 1; Table 2 (point IRRs) |
| 03 | `03_wild_cluster_bootstrap.py` | `r4_wildbootstrap_inference.csv` | §3.4 cross-check |
| 04 | `04_grid_inverted_wcr.py` | `r4_grid_inverted_ci.csv` (**intervals of record**) | Table 2 (CIs/p) |
| 05 | `05_fig_divergence.py` | `fig_divergence.png` | **Fig 1** |
| 06 | `06_wrong_vs_right.py` | `r3_wrong_vs_right.csv`, `fig_r3_wrong_vs_right.png` | Table (wrong-vs-right) |
| 09 | `09_simulation_instability.py` | `v2_instability_grid.csv`, `fig_v2_instability_surface.png` | **Fig 3**, §4.3 (Exp A) |
| 10 | `10_simulation_coverage.py` | `v2_coverage_table.csv`, `fig_v2_coverage.png` | **Fig 4**, §4.3 (Exp B) |
| 11 | `11_national_benchmark.py` | `v3_national_fars_series.csv`, `fig_v3_national_benchmark.png` | **Fig 5**, §4.4 |
| 12 | `12_figures_redesign.py` | `fig_v7_fig2_hybrid.png` | **Fig 2** |
| 13 | `13_worked_example.py` | `worked_example.csv` | Table 3 (county screen); target-setting arithmetic |
| 14 | `14_second_region.py` | `cd_second_region_irr.csv`, `cd_vs_national.csv`, `fig_v9_second_region.png` | §4.4 (G=4 floor, one sentence) |
| 15 | `15_third_region.py` | `njtpa_third_region_irr.csv`, `fig_njtpa.png` | **Fig 6**, §4.4 (featured replication) |
| 16 | `16_vmt_sensitivity.py` | `r11_vmt_sensitivity.csv` | §5.1 robustness |

Figure-number key: Fig 1 divergence · Fig 2 wrong-vs-right hybrid · Fig 3 instability surface ·
Fig 4 coverage bars · Fig 5 national benchmark · Fig 6 NJTPA replication.

---

## 5. Data dictionary (`./data`)

Inputs are a mix of author-derived aggregate CSVs and third-party tables; see `data/README.md`
for provenance and which files are shipped vs user-supplied. Counts are reportable crashes,
persons killed (KABCO "K", 30-day),
and persons seriously injured (KABCO "A"). Exposure is annual county VMT in hundreds of millions (`VMT_100M`).

| File | Role | Columns |
|---|---|---|
| `F_SI_C_Rates_Cleaned.csv` | NYMTC model input (counts + rates) | `County, Year, Cract` (crash count), `Crart` (crash rate /100M), `Kilct` (killed count), `Kilrt` (killed rate), `Serct` (serious count), `Serrt` (serious rate) |
| `nymtc_safety_panel.csv` | cleaned 60-row panel (regenerated by step 02) | `County, Year, crashes, fatalities, serious_injuries, VMT_100M, crash_rate_per100M, fatal_rate_per100M, serinj_rate_per100M` |
| `Draft_DVMT_and_Length_by_County_2016-2024_UAC_Update_1_.xlsx` | NYSDOT DVMT exposure source (DVMT × length, by functional class/urban area) — **not redistributed; user-supplied & optional** (see data/README.md §3) | workbook; reconstructed in step 01 if present, else documented |
| `njtpa_panel.csv` | NJTPA replication input (G=13) | `County, Year, crashes, killed, serious_injuries, VMT_100M` |
| `capital_district_rates.csv` | Capital District replication input (G=4) | same schema as `F_SI_C_Rates_Cleaned.csv` |
| `01_0108_…xlsx`, `05_0506_…xlsx` | all-crash and large-truck Day×Time×severity tables | workbooks (truck count-on-count demo) |
| `Contributing_Factors.xlsx`, `Crashes_With_{Human,Environmental,Vehicular}_Factors.xlsx` | factor tables (associational §5.3 only) | workbooks |

Selected regenerated outputs:

| File | Columns |
|---|---|
| `outputs/r4_grid_inverted_ci.csv` | `Outcome, Year, IRR, grid_CI_lo, grid_CI_hi, wcr_p_at_IRR1, contains_1, sig_test` |
| `outputs/worked_example.csv` | `county, crashes_2019, crashes_2024, crashes_pct, fatalities_2019, fatalities_2024, fatalities_pct, vmt_pct, crash_rate_RR, fatal_rate_RR, flip_counts_say_safer_rate_says_worse` |
| `outputs/njtpa_third_region_irr.csv` | `Outcome, Year, IRR, wcb_lo, wcb_hi, wcr_p` |

---

## 6. Reproducing the results

### Local
```
pip install -r requirements.lock.txt    # exact pinned env (use requirements.txt for loose)
python run_all.py                        # reads ./data, writes ./outputs, mirrors PNGs → ./figures
python tests/run_tests.py                # guards the core claims (run after run_all.py)
```
Python ≥ 3.11. Pure CPU; the full run is minutes on a laptop. `MPLBACKEND=Agg` is set in the container;
set it locally if running headless.

### Docker (pinned, self-verifying)
```
docker build -t nymtc-spf-repro .        # installs lockfile, then runs tests as the FINAL build step
docker run --rm -v "$PWD/outputs:/app/outputs" nymtc-spf-repro   # regenerate outputs to host
```
A failing `docker build` means a reproducibility/identification assertion broke.

### Continuous integration
`.github/workflows/ci.yml` runs on every push/PR/manual dispatch:
- **test** — `pip install -r requirements.lock.txt` then `python tests/run_tests.py` on Python 3.11 and 3.12;
- **docker** — `docker build` (which runs the suite on build);
- **lint** — `ruff` (informational; never fails the build).

---

## 7. Tests (what each asserts)

| Test | Guards |
|---|---|
| `test_enumeration.py` | the `2^10 = 1,024` wild-bootstrap sign-vector set is complete, unique, closed under negation |
| `test_provenance.py` | the provenance script reports the VMT ratio = 1.0000 identity and exits 0 — live (reconstructed from the user-supplied DVMT workbook) or documented (recorded result when the workbook is absent) |
| `test_bootstrap_ci.py` | grid-inverted CI/test consistency + regression check on 2021 fatality inference (IRR ≈ 1.33, CI ≈ [1.14, 1.58]) |
| `test_simulation_instability.py` | under a known-truth rare-event DGP, the conditional estimator is ~unbiased and tight while the count-on-count slope ratio is wide and sign-unstable |

---

## 8. Scope & limitations (read before reusing)

- **No vehicle-class exposure.** There is no truck VMT, so the truck/non-truck split appears **only** in
  the count-on-count counter-demonstration; the artifact forms **no** exposure-normalized truck risk and
  makes **no** causal claim about trucks. Closing this needs class-disaggregated exposure (WIM/HPMS/FHWA),
  not a different estimator on the same counts.
- **Few clusters.** G = 10 (NYMTC) / 13 (NJTPA) / 4 (Capital District). The bootstrap addresses inference;
  at G = 4 significance is unattainable by construction (floor `1/2^4 = 0.0625`). External validity beyond
  dense Northeastern metros over the pandemic window should be established by further replication.
- **Reduced-form, no covariates.** Identifies *that* per-mile risk rose, not *why*; §5.3 factor links are
  associational, not modeled.
- **Exposure is modeled DVMT,** not an odometer census; robustness to ±15% random / ±10% systematic offset
  error is shown in step 16.
- **Serious-injury coding** is KABCO "A" (NY) vs MMUCC "suspected serious injury" (NJ); serious-injury
  series are interpreted within-state, not compared across states.

---

## 9. Data sources & licensing

**Code** is released under the license in `LICENSE`. **Data files in `./data` carry their original
terms** — they are included for reproducibility, not relicensed. Provenance, checksums, and the
included-vs-user-supplied split are documented in `data/README.md`. In summary:

- Crash, fatality, and serious-injury counts: Institute for Traffic Safety Management and Research
  (ITSMR) Traffic Safety Statistical Repository — **publicly accessible; included.**
- NYMTC county VMT: the New York State DOT (NYSDOT) DVMT series — **not redistributed.**
  The DVMT workbook is user-supplied and optional; the provenance step (`src/01`) runs without it
  and records a re-verifiable result (ratio 1.0000) plus the source checksum (`data/README.md` §3).
- NJTPA crash + VMT: NJDOT crash records and NJDOT/HPMS county VMT, summarized in the
  author-derived `njtpa_panel.csv`.
- National benchmark: NHTSA FARS / FHWA travel volume (the national series is embedded in `src/11`).
- Author-derived county-year aggregate CSVs (`F_SI_C_Rates_Cleaned.csv`, `nymtc_safety_panel.csv`,
  `njtpa_panel.csv`, `capital_district_rates.csv`) are the authors' own compilations.

No third-party raw file is required to run the pipeline. To reproduce the method on another region,
swap the author-derived CSVs for your own county-year aggregates in the same schema (§5).

---

## 10. Versioning & citation

Version is set in `pyproject.toml` and `CITATION.cff`. On acceptance the repository will be archived with a
permanent DOI (Zenodo); cite the paper and the archived release. See `CITATION.cff` and the paper's
*Data and Code Availability* statement.

## 11. Troubleshooting

- **`FileNotFoundError` on a step** — run from the repo root (`run_all.py` and scripts use relative `data/`,
  `outputs/`). Run step 02 (and 04) before figure steps if invoking scripts individually; the figure scripts
  read regenerated CSVs.
- **Step 01 prints `[DOCUMENTED]` instead of `[LIVE]`** — expected: the NYSDOT DVMT exposure workbook is not shipped
  (not redistributed). The step reports the recorded provenance result and exits 0. To run the full live
  reconstruction, place the workbook in `data/` as described in `data/README.md` §3.
- **Matplotlib opens no window / errors headless** — set `MPLBACKEND=Agg`.
- **A figure looks different but numbers match** — figures are re-encoded from the result CSVs; styling
  (color/linestyle/hatch/dpi) is cosmetic and does not change reported numbers.
- **CI fails only on lint** — lint is informational and does not gate; check the `test` and `docker` jobs.

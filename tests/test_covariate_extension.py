"""Tests for R17 covariate extension (manuscript Section 4.5).
Asserts the 60-row factor-rate panel builds completely and the covariate-adjusted
year IRRs match the values reported in the paper, so the extension is reproducible."""
import os, subprocess, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _ensure_outputs():
    """Run the step if its outputs are not present yet."""
    need = ["covariate_panel.csv", "covariate_baseline_irr.csv", "covariate_adjusted_irr.csv"]
    if not all(os.path.exists(os.path.join(ROOT, "outputs", f)) for f in need):
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "17_covariate_extension.py")],
                       cwd=ROOT, check=True)


def test_covariate_panel_complete():
    _ensure_outputs()
    cov = pd.read_csv(os.path.join(ROOT, "outputs", "covariate_panel.csv"))
    assert len(cov) == 60, f"expected 60 county-years, got {len(cov)}"
    for col in ("log_speed_rate", "log_distract_rate"):
        assert cov[col].notna().all(), f"missing values in {col}"


def test_adjusted_irrs_match_paper():
    _ensure_outputs()
    adj = pd.read_csv(os.path.join(ROOT, "outputs", "covariate_adjusted_irr.csv")
                      ).set_index("Year")
    base = pd.read_csv(os.path.join(ROOT, "outputs", "covariate_baseline_irr.csv")
                       ).set_index("Year")
    # Section 4.5 headline values (tolerances allow tiny numerical drift across BLAS)
    assert abs(base.loc[2021, "IRR"] - 1.333) < 0.02
    assert abs(base.loc[2024, "IRR"] - 1.195) < 0.02
    assert abs(adj.loc[2021, "IRR"] - 1.282) < 0.03
    assert abs(adj.loc[2024, "IRR"] - 1.224) < 0.03
    # the elevation persists: adjusted 2021/2022/2024 remain significant, 2023 does not
    for yr in (2020, 2021, 2022, 2024):
        assert adj.loc[yr, "p_wcb"] < 0.05, f"adjusted year {yr} unexpectedly non-significant"
    assert adj.loc[2023, "p_wcb"] >= 0.05, "2023 unexpectedly significant after adjustment"

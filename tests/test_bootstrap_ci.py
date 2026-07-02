"""Confidence-interval / test consistency of the grid-inverted wild cluster bootstrap,
plus a regression check on the fatality inference of record."""
import pandas as pd, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CI=os.path.join(ROOT,"outputs","r4_grid_inverted_ci.csv")
def test_ci_test_consistency():
    d=pd.read_csv(CI)
    # 1.0 is inside the grid-inverted CI exactly when the restricted-test p-value >= 0.05
    for _,r in d.iterrows():
        inside=(r.grid_CI_lo<=1.0<=r.grid_CI_hi)
        assert bool(r.contains_1)==inside, f"CI/contains_1 mismatch: {r.Outcome} {r.Year}"
        assert (r.wcr_p_at_IRR1>=0.05)==inside, f"CI/test inconsistency: {r.Outcome} {r.Year}"
def test_fatality_regression():
    d=pd.read_csv(CI); f=d[(d.Outcome=="Fatalities")&(d.Year==2021)].iloc[0]
    assert abs(f.IRR-1.33)<0.03, f"2021 fatality IRR drifted: {f.IRR}"
    assert abs(f.grid_CI_lo-1.14)<0.05 and abs(f.grid_CI_hi-1.58)<0.06, "2021 fatality CI drifted"
if __name__=="__main__":
    test_ci_test_consistency(); test_fatality_regression(); print("PASS test_bootstrap_ci")

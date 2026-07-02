"""
R11 — Sensitivity of the recovered year effects to VMT (offset) measurement error.

The exposure series is a MODELED DVMT estimate, not a measured odometer census, so
a reviewer reasonably asks how much the headline fatality year-effects depend on it.
County fixed effects already absorb any county-CONSTANT exposure error; what remains
is within-county, over-time mismeasurement of VMT. We probe two regimes:

  (A) Random classical error: perturb each county-year log-VMT offset by
      eps ~ N(0, sigma^2) (multiplicative VMT error of ~sigma), refit, and record
      how the 2021-peak and 2024 fatality IRRs move over many draws.

  (B) Systematic single-year error: a uniform s% misstatement of one year's VMT is
      collinear with that year's dummy, so it shifts that year's IRR by ~1/(1+s)
      analytically. We report the implied band directly.

Poisson QMLE point estimates (identical mean model to the NB2 in src/02) suffice for
point-estimate sensitivity. numpy + pandas only.
"""
import numpy as np, pandas as pd
from numpy.linalg import pinv

RNG = np.random.default_rng(7)
df = pd.read_csv("data/nymtc_safety_panel.csv").sort_values(["County", "Year"]).reset_index(drop=True)
counties = sorted(df.County.unique()); years = sorted(df.Year.unique()); base = years[0]

def design(d):
    cols = {"const": np.ones(len(d))}
    for c in counties[1:]: cols[f"cty_{c}"] = (d.County == c).astype(float)
    for y in years[1:]:    cols[f"yr_{y}"] = (d.Year == y).astype(float)
    return pd.DataFrame(cols, index=d.index)

X = design(df).values; names = list(design(df).columns)
y = df["fatalities"].values.astype(float)
off0 = np.log(df["VMT_100M"].values)
ycol = {nm: j for j, nm in enumerate(names) if nm.startswith("yr_")}

def qmle(Xm, yv, off, iters=100, tol=1e-11):
    b = np.zeros(Xm.shape[1]); pos = yv[yv > 0]
    b[0] = np.log(pos.sum() / np.exp(off).sum()) if pos.sum() > 0 else 0.0
    mu = np.exp(np.clip(Xm @ b + off, -30, 30)); sn = np.linalg.norm(Xm.T @ (yv - mu))
    for _ in range(iters):
        step = pinv(Xm.T @ (Xm * mu[:, None])) @ (Xm.T @ (yv - mu)); s = 1.0; ok = False
        for _ls in range(50):
            cand = b + s * step; mu_c = np.exp(np.clip(Xm @ cand + off, -30, 30))
            sn_c = np.linalg.norm(Xm.T @ (yv - mu_c))
            if np.all(np.isfinite(mu_c)) and sn_c <= sn + 1e-10: ok = True; break
            s *= 0.5
        if not ok: break
        b, mu, sn = cand, mu_c, sn_c
        if np.max(np.abs(s * step)) < tol: break
    return b

b0 = qmle(X, y, off0)
base_irr = {int(nm[3:]): float(np.exp(b0[k])) for nm, k in ycol.items()}
print("Baseline fatality IRRs (vs 2019):")
for yr in sorted(base_irr): print(f"  {yr}: {base_irr[yr]:.3f}")

# ---------- (A) random classical multiplicative error ----------
NDRAW = 1000
print(f"\n(A) Random within-county-year VMT error, {NDRAW} draws/level:")
rows = []
for sigma in (0.05, 0.10, 0.15):
    irr21, irr24 = [], []
    for _ in range(NDRAW):
        off = off0 + RNG.normal(0, sigma, size=len(off0))
        b = qmle(X, y, off)
        irr21.append(np.exp(b[ycol["yr_2021"]])); irr24.append(np.exp(b[ycol["yr_2024"]]))
    irr21 = np.array(irr21); irr24 = np.array(irr24)
    rows.append(dict(sigma=sigma,
                     irr2021_mean=round(irr21.mean(), 3), irr2021_lo=round(np.quantile(irr21, .025), 3), irr2021_hi=round(np.quantile(irr21, .975), 3),
                     irr2024_mean=round(irr24.mean(), 3), irr2024_lo=round(np.quantile(irr24, .025), 3), irr2024_hi=round(np.quantile(irr24, .975), 3)))
    print(f"  sigma={sigma:.0%}: 2021 IRR {irr21.mean():.3f} [{np.quantile(irr21,.025):.3f},{np.quantile(irr21,.975):.3f}]"
          f" | 2024 IRR {irr24.mean():.3f} [{np.quantile(irr24,.025):.3f},{np.quantile(irr24,.975):.3f}]")
pd.DataFrame(rows).to_csv("outputs/r11_vmt_sensitivity.csv", index=False)

# ---------- (B) systematic single-year shift (analytic) ----------
print("\n(B) Systematic s% error in a single year's VMT shifts that year's IRR by 1/(1+s):")
for s in (0.05, 0.10):
    print(f"  s={s:.0%}: 2024 IRR {base_irr[2024]:.2f} -> [{base_irr[2024]/(1+s):.2f}, {base_irr[2024]/(1-s):.2f}]"
          f" | 2021 IRR {base_irr[2021]:.2f} -> [{base_irr[2021]/(1+s):.2f}, {base_irr[2021]/(1-s):.2f}]")
print("\nSaved outputs/r11_vmt_sensitivity.csv")

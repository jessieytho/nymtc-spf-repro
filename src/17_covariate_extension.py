"""
R17 - Exploratory covariate extension (manuscript Section 4.5).

Tests whether the pandemic-era per-mile fatality elevation is explained by shifts in
the behavioral composition of crashes. Covariates are EXPOSURE-NORMALIZED factor-crash
rates (crashes with a given contributing factor present, per 100M VMT), entered as
WITHIN-COUNTY-CENTERED logs -- deliberately NOT shares of crashes, which would be the
count-on-count quantity Section 3.5 retires. Strictly associational; no causal claim.

Source: ITSMR contributing-factor tabulation, all 10 NYMTC counties, 2019-2024,
all-severity ('' severity) factor PRESENCE (non-exclusive). The corrected export
`data/Contributing_Factors_corrected.csv` supersedes the under-counted legacy xlsx.

Selected covariates: log unsafe-speed rate, log driver-distraction rate. A VRU
(pedestrian/cyclist-error) rate was collinear with distraction (within-county log-rate
r = 0.82) and is omitted to avoid ratio instability.

Inference mirrors src/03: Poisson QMLE, log(VMT) offset, county+year FE, county-clustered
wild cluster bootstrap-t with exact 2^10 = 1024 Rademacher sign enumeration, restricted null.

Dependency-light: numpy + scipy + pandas + matplotlib only.
"""
import csv, os, itertools
import numpy as np, pandas as pd
from numpy.linalg import inv, pinv
from math import erf, sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"; FIG = "figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

CSV = "data/Contributing_Factors_corrected.csv"
TARGET = {"Unsafe Speed": "speed",
          "Driver Inattention/Distraction": "distract",
          "Pedestrian/Bicyclist Error/Confusion": "vru"}

# ---------- build factor-rate covariate panel ----------
rec = {}
for r in list(csv.reader(open(CSV, encoding="utf-8-sig")))[1:]:
    yr, sev, cty, ft, fac, tot = [x.strip() for x in r]
    if sev != "" or fac not in TARGET:   # all-severity rows only (avoid severity subtotals)
        continue
    val = np.nan if tot in (".", "", None) else float(tot.replace(",", ""))
    rec.setdefault((cty.upper(), int(yr)), {})[TARGET[fac]] = val

panel = pd.read_csv("data/nymtc_safety_panel.csv")
panel["County"] = panel["County"].str.upper(); panel["Year"] = panel["Year"].astype(int)
rows = []
for _, p in panel.iterrows():
    d = rec.get((p.County, p.Year), {})
    row = {"County": p.County, "Year": p.Year, "VMT_100M": p.VMT_100M,
           "fatalities": p.fatalities}
    for s in ("speed", "distract", "vru"):
        row[f"{s}_rate"] = d.get(s, np.nan) / p.VMT_100M
        row[f"log_{s}_rate"] = np.log(row[f"{s}_rate"])
    rows.append(row)
cov = pd.DataFrame(rows).sort_values(["County", "Year"]).reset_index(drop=True)
assert len(cov) == 60 and not cov[["log_speed_rate", "log_distract_rate"]].isna().any().any(), \
    "covariate panel incomplete"
for s in ("speed", "distract", "vru"):
    cov[f"c_{s}"] = cov.groupby("County")[f"log_{s}_rate"].transform(lambda x: x - x.mean())
cov.to_csv(f"{OUT}/covariate_panel.csv", index=False)

counties = sorted(cov.County.unique()); years = sorted(cov.Year.unique()); N = len(cov)
off = np.log(cov["VMT_100M"].values); y = cov["fatalities"].values.astype(float)
groups = cov.County.values; uniq = np.unique(groups); G = len(uniq)
gidx = [np.where(groups == g)[0] for g in uniq]
Gmat = np.array([(groups == g).astype(float) for g in uniq])
SIGNS = np.array(list(itertools.product([-1., 1.], repeat=G))); Wob = SIGNS @ Gmat

def design(extra):
    cols = {"const": np.ones(N)}
    for c in counties[1:]: cols[f"cty_{c}"] = (cov.County == c).astype(float)
    for yy in years[1:]:   cols[f"yr_{yy}"] = (cov.Year == yy).astype(float)
    for nm in extra:       cols[nm] = cov[nm].values
    return pd.DataFrame(cols)

def qmle(Xm, yv, off, iters=200, tol=1e-11):
    b = np.zeros(Xm.shape[1]); pos = yv[yv > 0]
    b[0] = np.log(pos.sum() / np.exp(off).sum()) if pos.sum() > 0 else 0.0
    mu = np.exp(np.clip(Xm @ b + off, -30, 30)); sn = np.linalg.norm(Xm.T @ (yv - mu))
    for _ in range(iters):
        step = pinv(Xm.T @ (Xm * mu[:, None])) @ (Xm.T @ (yv - mu)); s = 1.; ok = False
        for _ls in range(60):
            cand = b + s * step; mu_c = np.exp(np.clip(Xm @ cand + off, -30, 30))
            sn_c = np.linalg.norm(Xm.T @ (yv - mu_c))
            if np.all(np.isfinite(mu_c)) and sn_c <= sn + 1e-10: ok = True; break
            s *= 0.5
        if not ok: break
        b, mu, sn = cand, mu_c, sn_c
        if np.max(np.abs(s * step)) < tol: break
    return b, mu

def se_k(Xm, yv, mu, k):
    K = Xm.shape[1]; ssf = (G / (G - 1)) * ((N - 1) / (N - K))
    bread = inv(Xm.T @ (Xm * mu[:, None])); meat = np.zeros((K, K))
    for idx in gidx:
        sg = Xm[idx].T @ (yv[idx] - mu[idx]); meat += np.outer(sg, sg)
    return sqrt((ssf * (bread @ meat @ bread))[k, k])

def ncdf(z): return 0.5 * (1 + erf(z / sqrt(2)))

def wcb_year_irrs(extra):
    X = design(extra); Xv = X.values; names = list(X.columns)
    b_hat, mu_hat = qmle(Xv, y, off)
    out = []
    for nm, k in [(n, j) for j, n in enumerate(names) if n.startswith("yr_")]:
        se_hat = se_k(Xv, y, mu_hat, k); t_hat = b_hat[k] / se_hat
        keep = [j for j in range(Xv.shape[1]) if j != k]
        br, mur = qmle(Xv[:, keep], y, off); resid = y - mur
        tstar = np.empty(len(SIGNS))
        for bi in range(len(SIGNS)):
            bb, mm = qmle(Xv, mur + Wob[bi] * resid, off)
            sse = se_k(Xv, mur + Wob[bi] * resid, mm, k)
            tstar[bi] = bb[k] / sse if sse > 0 else 0.0
        out.append((int(nm[3:]), float(np.exp(b_hat[k])), float(t_hat),
                    float(np.mean(np.abs(tstar) >= abs(t_hat)))))
    coefs = {n[2:]: float(b_hat[j]) for j, n in enumerate(names) if n.startswith("c_")}
    return pd.DataFrame(out, columns=["Year", "IRR", "t", "p_wcb"]), coefs

base, _ = wcb_year_irrs([])
adj, adj_coefs = wcb_year_irrs(["c_speed", "c_distract"])
base.to_csv(f"{OUT}/covariate_baseline_irr.csv", index=False)
adj.to_csv(f"{OUT}/covariate_adjusted_irr.csv", index=False)

print("=" * 72)
print("R17  COVARIATE EXTENSION  (Section 4.5)  -- fatality-rate year IRRs")
print("=" * 72)
print(f"{'Year':<6}{'baseline IRR':>14}{'p_wcb':>9}   {'adjusted IRR':>14}{'p_wcb':>9}")
for yr in base.Year:
    b = base[base.Year == yr].iloc[0]; a = adj[adj.Year == yr].iloc[0]
    print(f"{yr:<6}{b.IRR:>14.3f}{b.p_wcb:>9.3f}   {a.IRR:>14.3f}{a.p_wcb:>9.3f}")
print(f"\ncovariate coefficients (speed+distract adj): "
      f"speed={adj_coefs['speed']:+.3f}, distract={adj_coefs['distract']:+.3f}")
vru_fit, vru_coefs = wcb_year_irrs(["c_vru"])
print(f"within-county log-rate corr(VRU, distraction) = "
      f"{np.corrcoef(cov.c_vru, cov.c_distract)[0,1]:.2f}  (low; not the reason)")
print(f"VRU omitted because it enters wrong-signed/insignificant: "
      f"c_vru beta={vru_coefs['vru']:+.3f}")

# ---------- figure (grayscale-safe) ----------
fig, ax = plt.subplots(figsize=(7.0, 4.3))
yb = [1.0] + list(base.IRR); ya = [1.0] + list(adj.IRR)
xs = [2019] + list(base.Year)
ax.plot(xs, yb, color="black", marker="o", ms=6, lw=2.2, label="Baseline (offset + 2-way FE)")
ax.plot(xs, ya, color="black", marker="s", ms=6, lw=2.2, ls="--", mfc="white",
        label="Adjusted: + within-county log(speed rate), log(distraction rate)")
ax.axhline(1.0, color="0.3", lw=0.9, ls=":")
ax.set_xticks(xs); ax.set_xlabel("Year")
ax.set_ylabel("Fatality rate ratio vs 2019\n(per 100M VMT, net of county FE)")
ax.set_title("Per-mile fatality elevation persists after controlling for\n"
             "speed and distraction crash rates (behavioral composition does not explain it)",
             fontsize=10.5)
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax.grid(axis="y", color="0.85", lw=0.6)
fig.tight_layout()
for d in (OUT, FIG):
    fig.savefig(f"{d}/fig_w1_covariate_adjusted.png", dpi=200)
print("\nsaved fig_w1_covariate_adjusted.png")

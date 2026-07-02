"""
R9 (featured) — Second-region replication: NJTPA (G=13), the properly powered
external replication used in Section 4.4.

Runs the SAME SPF-offset Poisson-QMLE estimator and wild-cluster-bootstrap-t
inference used for NYMTC (src/02-04) on the thirteen counties of the North
Jersey Transportation Planning Authority, 2019-2024.

Data: data/njtpa_panel.csv  (County, Year, crashes, killed, serious_injuries, VMT_100M)
  - crashes  = total reportable crashes (crash-level), NJDOT crash dashboard,
    by county and year.
  - killed   = persons killed (K, person-level; NJDOT "Total Killed", validated
    against the person-level injury arrays).
  - serious_injuries = persons with a suspected serious injury (A, person-level;
    NJDOT "Total Incapacitated"). NJ codes severity under MMUCC, so this series
    is interpreted WITHIN New Jersey only (as the NYMTC "A" series is within NY).
  - VMT_100M = annual county VMT in 100M units, from the NJDOT/HPMS "Roadway
    Mileage and Daily VMT by Functional Classification Distributed by County"
    files (daily VMT x days/year / 1e8); statewide totals validated to NJDOT's
    published figures, NJTPA-13 ~72% of state every year.

With G=13 the exact wild-bootstrap enumeration is 2^13 = 8,192 sign vectors;
unlike the four-county Capital District (src/14, G=4, floor p=0.0625), the test
HAS power: crash rate down and fatality rate up are both significant in 2020-21.

numpy + scipy + pandas + matplotlib only.
"""
import numpy as np, pandas as pd, itertools, os
from numpy.linalg import inv, pinv
from math import erf, sqrt

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv("data/njtpa_panel.csv").sort_values(["County", "Year"]).reset_index(drop=True)
counties = sorted(df.County.unique()); years = sorted(df.Year.unique()); base = years[0]

def design(d):
    cols = {"const": np.ones(len(d))}
    for c in counties[1:]: cols[f"cty_{c}"] = (d.County == c).astype(float)
    for y in years[1:]:    cols[f"yr_{y}"]  = (d.Year == y).astype(float)
    return pd.DataFrame(cols, index=d.index)

X = design(df); names = list(X.columns); Xv = X.values
off = np.log(df.VMT_100M.values); groups = df.County.values
uniq = np.unique(groups); G = len(uniq); N, K = Xv.shape
ycols = {nm: j for j, nm in enumerate(names) if nm.startswith("yr_")}
Gmat = np.array([(groups == g).astype(float) for g in uniq])
gidx = [np.where(groups == g)[0] for g in uniq]
ss = (G / (G - 1)) * ((N - 1) / (N - K))
SIGNS = np.array(list(itertools.product([-1.0, 1.0], repeat=G))); Wobs = SIGNS @ Gmat
print(f"NJTPA  G={G}, N={N}, K={K}; exact wild-bootstrap vectors 2^{G}={len(SIGNS)}; min two-sided WCR p={1/len(SIGNS):.5f}")

def qmle(Xm, y, off, iters=100, tol=1e-11):
    b = np.zeros(Xm.shape[1]); pos = y[y > 0]
    b[0] = np.log(pos.sum() / np.exp(off).sum()) if pos.sum() > 0 else 0.0
    mu = np.exp(np.clip(Xm @ b + off, -30, 30)); sn = np.linalg.norm(Xm.T @ (y - mu))
    for _ in range(iters):
        step = pinv(Xm.T @ (Xm * mu[:, None])) @ (Xm.T @ (y - mu)); s = 1.0; ok = False
        for _ls in range(50):
            cand = b + s * step; mu_c = np.exp(np.clip(Xm @ cand + off, -30, 30))
            sn_c = np.linalg.norm(Xm.T @ (y - mu_c))
            if np.all(np.isfinite(mu_c)) and sn_c <= sn + 1e-10: ok = True; break
            s *= 0.5
        if not ok: break
        b, mu, sn = cand, mu_c, sn_c
        if np.max(np.abs(s * step)) < tol: break
    return b, mu

def se_cl(Xm, y, mu, k):
    bread = inv(Xm.T @ (Xm * mu[:, None])); meat = np.zeros((Xm.shape[1],) * 2)
    for idx in gidx:
        sg = Xm[idx].T @ (y[idx] - mu[idx]); meat += np.outer(sg, sg)
    return sqrt((ss * (bread @ meat @ bread))[k, k])

nat = pd.DataFrame({"Year": [2019, 2020, 2021, 2022, 2023, 2024], "rate": [1.11, 1.34, 1.38, 1.34, 1.26, 1.20]})
nat["rr"] = nat.rate / nat.rate.iloc[0]

def run(label, col):
    y = df[col].values.astype(float)
    b, mu = qmle(Xv, y, off); pear = np.sum((y - mu) ** 2 / mu) / (N - K)
    resid = y - mu
    bstar = np.full((len(SIGNS), K), np.nan); sestar = np.full((len(SIGNS), K), np.nan)
    for bi in range(len(SIGNS)):
        ys = mu + Wobs[bi] * resid; bb, mm = qmle(Xv, ys, off); bstar[bi] = bb
        for nm, k in ycols.items(): sestar[bi, k] = se_cl(Xv, ys, mm, k)
    rows = []
    for nm, k in ycols.items():
        yr = int(nm[3:]); irr = float(np.exp(b[k])); se = se_cl(Xv, y, mu, k); t = b[k] / se
        tci = (bstar[:, k] - b[k]) / sestar[:, k]; qlo, qhi = np.quantile(tci, [.025, .975])
        ci_lo = float(np.exp(b[k] - qhi * se)); ci_hi = float(np.exp(b[k] - qlo * se))
        keep = [j for j in range(K) if j != k]; brs, _ = qmle(Xv[:, keep], y, off)
        br = np.zeros(K); br[keep] = brs; mur = np.exp(np.clip(Xv @ br + off, -30, 30)); rr_ = y - mur; cnt = 0
        for bi in range(len(SIGNS)):
            ys = mur + Wobs[bi] * rr_; bb, mm = qmle(Xv, ys, off); seb = se_cl(Xv, ys, mm, k)
            if abs(bb[k] / seb) >= abs(t): cnt += 1
        rows.append(dict(Outcome=label, Year=yr, IRR=round(irr, 3), wcb_lo=round(ci_lo, 3),
                         wcb_hi=round(ci_hi, 3), wcr_p=round(cnt / len(SIGNS), 4)))
    t = pd.DataFrame(rows)
    print(f"\n### {label}  (Pearson dispersion={pear:.2f})")
    for _, r in t.iterrows():
        print(f"  {r.Year}  IRR={r.IRR:5.3f}  WCB-t[{r.wcb_lo:.3f},{r.wcb_hi:.3f}]  WCR_p={r.wcr_p:.4f}"
              f"{'*' if r.wcr_p < .05 else ''}")
    return t

tabs = {lab: run(lab, col) for lab, col in
        [("Crashes", "crashes"), ("Fatalities", "killed"), ("Serious injuries", "serious_injuries")]}
fat = tabs["Fatalities"].sort_values("Year")
r_shape = float(np.corrcoef(fat.IRR.values, nat.rr.values[1:])[0, 1])
print(f"\nPearson r (NJTPA fatality IRR vs national 2020-24) = {r_shape:.3f}  [NYMTC 0.95, Capital District -0.27]")

# ---- figure (fig:njtpa) ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cra = tabs["Crashes"].sort_values("Year"); ser = tabs["Serious injuries"].sort_values("Year")
yrs = [base] + list(fat.Year)
fa = [1.0] + list(fat.IRR); fl = [1.0] + list(fat.wcb_lo); fh = [1.0] + list(fat.wcb_hi)
cr = [1.0] + list(cra.IRR); cl = [1.0] + list(cra.wcb_lo); ch = [1.0] + list(cra.wcb_hi)
se = [1.0] + list(ser.IRR)
fig, ax = plt.subplots(figsize=(7.8, 4.8))
ax.fill_between(yrs, fl, fh, color="#c53030", alpha=0.12, zorder=1)
ax.fill_between(yrs, cl, ch, color="#2b6cb0", alpha=0.12, zorder=1)
ax.plot(yrs, fa, marker="s", color="#c53030", lw=2.3, zorder=4, label="Fatality rate ratio (model IRR, 95% WCB-t band)")
ax.plot(yrs, cr, marker="o", color="#2b6cb0", lw=2.3, zorder=4, label="Crash rate ratio (model IRR, 95% WCB-t band)")
ax.plot(nat.Year, nat.rr, marker="D", ms=4, color="#1f4e79", lw=1.8, ls="--", zorder=3, label="National FARS/FHWA fatality rate ratio")
ax.plot(yrs, se, marker="^", ms=5, color="#2f855a", lw=1.5, ls=":", zorder=2, label="Serious-injury rate ratio (model IRR)")
ax.axhline(1.0, color="#666", lw=1, ls=":")
ax.set_xticks(yrs); ax.set_ylabel("Rate ratio vs 2019"); ax.set_xlabel("Year")
ax.set_title(f"Second-region replication (NJTPA, G={G}): emptier but deadlier, with real power.\n"
             f"Crash rate falls significantly in 2020-21 while the fatality rate rises significantly\n"
             f"and tracks the national series (r={r_shape:.2f})", fontsize=9.5, weight="bold")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_njtpa.png", dpi=200, bbox_inches="tight")
pd.concat(tabs.values(), ignore_index=True).to_csv(f"{OUT}/njtpa_third_region_irr.csv", index=False)
print(f"\nSaved {OUT}/fig_njtpa.png, {OUT}/njtpa_third_region_irr.csv")

"""
R9 - Second-region replication: Capital District (CDTC) transferability check.

Runs the SAME SPF-offset Poisson-QMLE estimator (log VMT offset, county+year FE)
and wild-cluster-bootstrap-t inference used for NYMTC (src/02-04), on a second,
structurally different region: the four core Capital District counties
(Albany, Rensselaer, Saratoga, Schenectady), 2019-2024.

Data: data/capital_district_rates.csv  (same schema as F_SI_C_Rates_Cleaned.csv)
  - crash / persons-killed (K) / persons-serious (A) counts from ITSMR TSSR
    (County Crash Summary + Persons Killed or Injured), same definitions as NYMTC;
    Schenectady-2022 crash count = 4,069 (Crash Type Summary; the County Crash
    Summary export for that one cell was truncated to 803).
  - VMT from the NYSDOT DVMT workbook 'VMT by County' sheet, annualized exactly
    as src/01 (validated: reproduces the NYMTC VMT to ratio 1.00000).

KEY INFERENTIAL CAVEAT (handled honestly): with G=4 clusters the wild cluster
bootstrap enumerates only 2^4 = 16 sign vectors, so the minimum attainable
two-sided WCR p-value is 1/16 = 0.0625 and grid-inverted WCR intervals are
unbounded. This region is therefore a POINT-ESTIMATE CONSISTENCY check: does the
recovered fatality-rate trajectory track the external national FARS series (the
same benchmark used for NYMTC, r=0.95)? Significance is not, and cannot be,
claimed at G=4 -- which is itself the paper's few-cluster point.

numpy + scipy + pandas + matplotlib only.
"""
import numpy as np, pandas as pd, itertools, os
from numpy.linalg import inv, pinv
from math import erf, sqrt

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("data/capital_district_rates.csv").sort_values(["County", "Year"]).reset_index(drop=True)
df["VMT_100M"] = df["Cract"] / df["Crart"]
counties = sorted(df.County.unique()); years = sorted(df.Year.unique())
base_year = years[0]

def design(d):
    cols = {"const": np.ones(len(d))}
    for c in counties[1:]: cols[f"cty_{c}"] = (d.County == c).astype(float)
    for y in years[1:]:    cols[f"yr_{y}"]  = (d.Year == y).astype(float)
    return pd.DataFrame(cols, index=d.index)

X = design(df); names = list(X.columns); Xv = X.values
off = np.log(df["VMT_100M"].values); groups = df.County.values
uniq = np.unique(groups); G = len(uniq); N, K = Xv.shape
year_cols = {nm: j for j, nm in enumerate(names) if nm.startswith("yr_")}
Gmat = np.array([(groups == g).astype(float) for g in uniq])
gidx = [np.where(groups == g)[0] for g in uniq]
ss = (G / (G - 1)) * ((N - 1) / (N - K))
SIGNS = np.array(list(itertools.product([-1.0, 1.0], repeat=G))); Wobs = SIGNS @ Gmat

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

def se_cluster(Xm, y, mu, k):
    bread = inv(Xm.T @ (Xm * mu[:, None])); meat = np.zeros((Xm.shape[1],) * 2)
    for idx in gidx:
        sg = Xm[idx].T @ (y[idx] - mu[idx]); meat += np.outer(sg, sg)
    return sqrt((ss * (bread @ meat @ bread))[k, k])

def ncdf(z): return 0.5 * (1 + erf(z / sqrt(2)))

def infer(label, col):
    y = df[col].values.astype(float)
    b_hat, mu_hat = qmle(Xv, y, off)
    pearson = np.sum((y - mu_hat) ** 2 / mu_hat) / (N - K)
    rows = []
    for nm, k in year_cols.items():
        yr = int(nm[3:]); irr = float(np.exp(b_hat[k]))
        se_k = se_cluster(Xv, y, mu_hat, k); t_obs = b_hat[k] / se_k
        # restricted WCR p (exact 16 vectors) -- demonstrates the G=4 floor
        keep = [j for j in range(K) if j != k]
        b_r_sub, _ = qmle(Xv[:, keep], y, off); b_r = np.zeros(K); b_r[keep] = b_r_sub
        mu_r = np.exp(np.clip(Xv @ b_r + off, -30, 30)); resid_r = y - mu_r
        cnt = 0
        for bi in range(len(SIGNS)):
            ys = mu_r + Wobs[bi] * resid_r
            bb, mm = qmle(Xv, ys, off); se_b = se_cluster(Xv, ys, mm, k)
            if abs(bb[k] / se_b) >= abs(t_obs): cnt += 1
        rows.append(dict(Outcome=label, Year=yr, IRR=round(irr, 4),
                         analytic_CI_lo=round(float(np.exp(b_hat[k] - 1.96*se_k)), 4),
                         analytic_CI_hi=round(float(np.exp(b_hat[k] + 1.96*se_k)), 4),
                         wcr_p_exact16=round(cnt/len(SIGNS), 4)))
    return pearson, pd.DataFrame(rows)

print("=" * 84)
print(f"R9  SECOND-REGION REPLICATION - Capital District (G={G} counties, T={len(years)}, N={N})")
print(f"    Exact wild-bootstrap enumeration: 2^{G} = {len(SIGNS)} sign vectors; min WCR p = {1/len(SIGNS):.4f}")
print("=" * 84)
tables = {}
for label, col in [("Crashes", "Cract"), ("Fatalities", "Kilct"), ("Serious injuries", "Serct")]:
    pe, t = infer(label, col); tables[label] = t
    print(f"\n### {label}  (Poisson Pearson dispersion = {pe:.2f})")
    print("  Yr     IRR    analytic95[lo,hi]   WCR_p(exact16)")
    for _, r in t.iterrows():
        print(f"  {r.Year}  {r.IRR:5.3f}   [{r.analytic_CI_lo:4.2f}, {r.analytic_CI_hi:4.2f}]      {r.wcr_p_exact16:.4f}")

# ---- external benchmark: CD fatality IRRs vs national FARS rate-ratio series ----
nat = pd.DataFrame({"Year": [2019,2020,2021,2022,2023,2024],
                    "rate_per100M": [1.11,1.34,1.38,1.34,1.26,1.20]})
nat["rate_ratio_2019"] = nat.rate_per100M / nat.rate_per100M.iloc[0]
fat = tables["Fatalities"].sort_values("Year")
cd_irr = np.array(list(fat.IRR)); natr = nat.rate_ratio_2019.values[1:]
r_shape = float(np.corrcoef(cd_irr, natr)[0, 1])
cra = tables["Crashes"].sort_values("Year")
print("\n" + "-" * 84)
print("EXTERNAL CONSISTENCY (point-estimate; no significance claim at G=4)")
print(f"  National fatality rate ratio vs 2019 : {[round(x,3) for x in nat.rate_ratio_2019]}")
print(f"  CD     fatality IRR      vs 2019     : {[1.0]+[round(x,3) for x in cd_irr]}")
print(f"  CD     crash    IRR      vs 2019     : {[1.0]+[round(x,3) for x in cra.IRR]}")
print(f"  Pearson r (CD fatality vs national, 2020-2024) = {r_shape:.3f}")
print(f"  CD 2024 fatality IRR = {cd_irr[-1]:.3f}; CD 2024 crash IRR = {cra.IRR.iloc[-1]:.3f}")

# ---- figure mirroring fig:benchmark ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
yrs = [base_year] + list(fat.Year); irr = [1.0] + list(fat.IRR)
fig, ax = plt.subplots(figsize=(7.6, 4.7))
ax.plot(yrs, irr, marker="s", color="#c53030", lw=2.2, zorder=3,
        label="Capital District fatality rate ratio (model IRR)")
ax.plot([base_year]+list(cra.Year), [1.0]+list(cra.IRR), marker="^", color="#2b6cb0",
        lw=1.8, ls="--", zorder=2, label="Capital District crash rate ratio (model IRR)")
ax.plot(nat.Year, nat.rate_ratio_2019, marker="o", color="#1f4e79", lw=2.2, ls="--", zorder=3,
        label="National FARS/FHWA fatality rate ratio")
ax.axhline(1.0, color="#666", lw=1, ls=":")
ax.set_xticks(yrs); ax.set_ylabel("Rate ratio vs 2019"); ax.set_xlabel("Year")
ax.set_title("Second-region replication (Capital District, G=4): per-mile lethality is\n"
             f"elevated (2024 fatality RR {cd_irr[-1]:.2f}) while the crash rate stays flat;\n"
             f"the noisy year-to-year fatality path does not track national shape (r={r_shape:.2f})",
             fontsize=9.5, weight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_v9_second_region.png", dpi=200, bbox_inches="tight")

# ---- persist ----
allt = pd.concat(tables.values(), ignore_index=True)
allt.to_csv(f"{OUT}/cd_second_region_irr.csv", index=False)
pd.DataFrame({"Year": nat.Year, "national_rr": nat.rate_ratio_2019,
              "cd_fatality_irr": [1.0]+list(cd_irr),
              "cd_crash_irr": [1.0]+list(cra.IRR)}).round(4).to_csv(f"{OUT}/cd_vs_national.csv", index=False)
print("\nSaved: outputs/fig_v9_second_region.png, cd_second_region_irr.csv, cd_vs_national.csv")
print(f"[note] G={G} -> min attainable two-sided WCR p = 1/2^{G} = {1/len(SIGNS):.4f}; "
      f"point estimates + external consistency carry the replication.")

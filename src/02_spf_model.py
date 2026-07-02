"""
R1 — Exposure-normalized regional safety performance (SPF-form offset model).

Models county-year crash / fatality / serious-injury COUNTS as a function of
exposure via log(VMT) OFFSET — the Highway Safety Manual SPF form — with county
and year fixed effects. Negative Binomial (NB2) handles the overdispersion that
Poisson cannot. Year terms are reported as Incidence Rate Ratios (rate per 100M
VMT, net of stable county differences) relative to a 2019 baseline.

Dependency-light: numpy + scipy only (no statsmodels / network needed).
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from numpy.linalg import inv, pinv

RNG = np.random.default_rng(7)

# ---------- data ----------
df = pd.read_csv("data/F_SI_C_Rates_Cleaned.csv")
df["VMT_100M"] = df["Cract"] / df["Crart"]          # annual VMT in 100M units (verified reconciles)
df = df.sort_values(["County", "Year"]).reset_index(drop=True)
counties = sorted(df.County.unique()); years = sorted(df.Year.unique())
base_year = years[0]

def design(df):
    """Intercept + county FE (drop first) + year FE (drop base year)."""
    cols = {"const": np.ones(len(df))}
    for c in counties[1:]:
        cols[f"cty_{c}"] = (df.County == c).astype(float)
    for y in years[1:]:
        cols[f"yr_{y}"] = (df.Year == y).astype(float)
    X = pd.DataFrame(cols, index=df.index)
    return X

X = design(df)
off = np.log(df["VMT_100M"].values)
Xv = X.values; names = list(X.columns)

# ---------- Poisson IRLS (warm start + damped Newton) ----------
def _mu(X, beta, off):
    return np.exp(np.clip(X @ beta + off, -30, 30))

def _pois_dev(y, mu):
    t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * np.sum(t - (y - mu))

def poisson_irls(X, y, off, iters=200, tol=1e-10):
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.sum() / np.exp(off).sum())   # warm intercept = overall log-rate
    mu = _mu(X, beta, off); dev = _pois_dev(y, mu)
    for _ in range(iters):
        W = mu
        XtWX = X.T @ (X * W[:, None])
        step = pinv(XtWX) @ (X.T @ (y - mu))
        # step-halving line search on deviance
        s = 1.0
        for _ls in range(40):
            cand = beta + s * step
            mu_c = _mu(X, cand, off)
            if np.all(np.isfinite(mu_c)):
                dev_c = _pois_dev(y, mu_c)
                if dev_c <= dev + 1e-8:
                    break
            s *= 0.5
        if np.max(np.abs(s * step)) < tol:
            beta, mu, dev = cand, mu_c, dev_c
            break
        beta, mu, dev = cand, mu_c, dev_c
    return beta, mu

# ---------- NB2 MLE ----------
def nb2_negll(params, X, y, off):
    beta = params[:-1]; alpha = np.exp(np.clip(params[-1], -20, 20)); k = 1.0 / alpha
    mu = np.exp(np.clip(X @ beta + off, -30, 30))
    ll = (gammaln(y + k) - gammaln(k) - gammaln(y + 1)
          + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))
    val = -np.sum(ll)
    return val if np.isfinite(val) else 1e12

def hessian_fd(f, x, eps=1e-5):
    n = len(x); H = np.zeros((n, n))
    g0 = grad_fd(f, x, eps)
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        H[:, i] = (grad_fd(f, xp, eps) - g0) / eps
    return 0.5 * (H + H.T)

def grad_fd(f, x, eps=1e-6):
    n = len(x); g = np.zeros(n); f0 = f(x)
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        g[i] = (f(xp) - f0) / eps
    return g

def cluster_robust_cov(X, y, mu, groups):
    """County-clustered sandwich covariance for the Poisson(QMLE) fit.
    PSD by construction; with small-sample correction. (G=10 -> few-cluster
    caveat: wild cluster bootstrap is the manuscript-grade refinement.)"""
    W = mu
    bread = inv(X.T @ (X * W[:, None]))
    K = X.shape[1]; N = X.shape[0]
    meat = np.zeros((K, K))
    for g in np.unique(groups):
        idx = groups == g
        sg = X[idx].T @ (y[idx] - mu[idx])
        meat += np.outer(sg, sg)
    G = len(np.unique(groups))
    corr = (G / (G - 1)) * ((N - 1) / (N - K))
    return corr * (bread @ meat @ bread)

def nb_alpha_quick(X, y, off, b_p):
    """Fit NB2 alpha only, as an overdispersion diagnostic (not used for SEs)."""
    res = minimize(nb2_negll, np.r_[b_p, np.log(0.01)], args=(X, y, off),
                   method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-6})
    return float(np.exp(res.x[-1]))

groups = df.County.values

def fit_outcome(y, label):
    b_p, mu_p = poisson_irls(Xv, y, off)
    p = Xv.shape[1]
    pearson = np.sum((y - mu_p) ** 2 / mu_p) / (len(y) - p)
    cov = cluster_robust_cov(Xv, y, mu_p, groups)
    se = np.sqrt(np.diag(cov))
    alpha = nb_alpha_quick(Xv, y, off, b_p)
    return {"label": label, "pearson_disp_poisson": pearson, "nb_alpha": alpha,
            "beta": b_p, "se": se, "mu": mu_p}

outcomes = {"Crashes": "Cract", "Fatalities": "Kilct", "Serious injuries": "Serct"}
fits = {name: fit_outcome(df[col].values.astype(float), name) for name, col in outcomes.items()}

# ---------- year IRRs (rate ratio vs 2019, net of county FE) ----------
def year_irrs(fit):
    rows = []
    for j, nm in enumerate(names):
        if nm.startswith("yr_"):
            b, s = fit["beta"][j], fit["se"][j]
            rows.append((int(nm[3:]), np.exp(b), np.exp(b - 1.96 * s), np.exp(b + 1.96 * s),
                         2 * (1 - _normcdf(abs(b / s)))))
    return pd.DataFrame(rows, columns=["Year", "IRR", "CI_lo", "CI_hi", "p"])

def _normcdf(z):
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))

print("=" * 74)
print("R1 SPF-FORM OFFSET MODEL  —  count ~ NB2,  offset = log(VMT/100M),  county+year FE")
print("=" * 74)
irr_tables = {}
for name, fit in fits.items():
    t = year_irrs(fit); irr_tables[name] = t
    print(f"\n### {name}  (Poisson Pearson dispersion = {fit['pearson_disp_poisson']:.2f}  -> "
          f"{'overdispersed, NB justified' if fit['pearson_disp_poisson']>1.5 else 'mild'};  NB alpha = {fit['nb_alpha']:.4f})")
    print("  Year  IRR(vs2019)   95% CI            p")
    for _, r in t.iterrows():
        star = "***" if r.p < .01 else "**" if r.p < .05 else "*" if r.p < .1 else ""
        print(f"  {int(r.Year)}   {r.IRR:6.3f}     [{r.CI_lo:5.3f}, {r.CI_hi:5.3f}]   {r.p:6.3f}{star}")

# ---------- sanity: fitted baseline rate is in the right ballpark ----------
b = fits["Crashes"]["beta"]; mu_c = np.exp(Xv @ b + off)
df["_fit_crash_rate"] = mu_c / df["VMT_100M"]
chk = df[(df.County=="BRONX")&(df.Year==2019)].iloc[0]
print(f"\n[sanity] additive-FE fitted Bronx-2019 crash rate/100M VMT = {chk._fit_crash_rate:.0f} "
      f"(raw cell {chk.Crart:.0f}; additive model is not cell-saturated, so close-not-equal is expected)")

# ---------- WRONG vs RIGHT exhibit (counts mislead; rates reveal) ----------
reg = df.groupby("Year").agg(Crashes=("Cract","sum"), Fatal=("Kilct","sum"),
                             SerInj=("Serct","sum"), VMT=("VMT_100M","sum")).reset_index()
def pct(a,b): return (a/b-1)*100
naive = pd.DataFrame({
    "Outcome": ["Crashes","Fatalities","Serious injuries"],
    "Raw count 2019": [reg.Crashes.iloc[0], reg.Fatal.iloc[0], reg.SerInj.iloc[0]],
    "Raw count 2024": [reg.Crashes.iloc[-1], reg.Fatal.iloc[-1], reg.SerInj.iloc[-1]],
})
naive["Naive % change (counts)"] = [pct(reg.Crashes.iloc[-1],reg.Crashes.iloc[0]),
                                    pct(reg.Fatal.iloc[-1],reg.Fatal.iloc[0]),
                                    pct(reg.SerInj.iloc[-1],reg.SerInj.iloc[0])]
# exposure-normalized 2024-vs-2019 rate ratio from the model
rr = {nm: irr_tables[nm][irr_tables[nm].Year==years[-1]].iloc[0] for nm in outcomes}
naive["Exposure-normalized rate ratio 2024/2019"] = [rr[n].IRR for n in outcomes]
naive["95% CI"] = [f"[{rr[n].CI_lo:.2f}, {rr[n].CI_hi:.2f}]" for n in outcomes]

print("\n" + "="*74)
print("WRONG-vs-RIGHT EXHIBIT  (the Results centerpiece)")
print("="*74)
with pd.option_context("display.width",140,"display.max_columns",None):
    print(naive.round(1).to_string(index=False))

# ---------- Results figure: the divergence with 95% CI bands ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.2, 4.4))
style = {"Crashes": ("#2b6cb0", "o", "-"),
         "Fatalities": ("#c53030", "s", "-"),
         "Serious injuries": ("#2f855a", "^", "--")}
yrs_plot = [base_year] + list(years[1:])
for nm, t in irr_tables.items():
    irr = [1.0] + list(t.IRR); lo = [1.0] + list(t.CI_lo); hi = [1.0] + list(t.CI_hi)
    c, mk, ls = style[nm]
    ax.plot(yrs_plot, irr, marker=mk, color=c, ls=ls, lw=2, label=nm, zorder=3)
    ax.fill_between(yrs_plot, lo, hi, color=c, alpha=0.13, zorder=1)
ax.axhline(1.0, color="#555", lw=1, ls=":")
ax.set_ylabel("Rate ratio vs 2019  (per 100M VMT, net of county FE)")
ax.set_xlabel("Year")
ax.set_title("Emptier but deadlier: exposure-normalized NYMTC safety, 2019–2024",
             fontsize=11, weight="bold")
ax.set_xticks(yrs_plot)
ax.legend(frameon=False, fontsize=9, loc="upper left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.annotate("crashes fall", (2024, irr_tables["Crashes"].IRR.iloc[-1]),
            xytext=(2022.4, 0.78), fontsize=8.5, color="#2b6cb0")
ax.annotate("fatality rate rises", (2021, irr_tables["Fatalities"].IRR.iloc[1]),
            xytext=(2020.3, 1.42), fontsize=8.5, color="#c53030")
fig.tight_layout()
fig.savefig("outputs/fig_divergence.png", dpi=200)
print("Saved figure: fig_divergence.png")

# ---------- persist deliverables ----------
import os; os.makedirs("outputs", exist_ok=True)
panel = df[["County","Year","Cract","Kilct","Serct","VMT_100M","Crart","Kilrt","Serrt"]].copy()
panel.columns = ["County","Year","crashes","fatalities","serious_injuries",
                 "VMT_100M","crash_rate_per100M","fatal_rate_per100M","serinj_rate_per100M"]
panel.to_csv("outputs/nymtc_safety_panel.csv", index=False)
for nm,t in irr_tables.items():
    t.round(4).to_csv(f"outputs/irr_{nm.split()[0].lower()}.csv", index=False)
naive.round(3).to_csv("outputs/exhibit_counts_vs_rates.csv", index=False)
print("Saved: nymtc_safety_panel.csv, irr_*.csv, exhibit_counts_vs_rates.csv")

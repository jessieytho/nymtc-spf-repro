"""
R4b - Grid-inverted WCR confidence intervals (full CI<->test consistency).

Inverts the restricted (WCR) wild-cluster bootstrap test: the CI is the set of
null IRRs the test fails to reject at 5%. By construction IRR=1 lies in the CI
iff the WCR p-value at IRR=1 is >= 0.05, so interval and test always agree.

For a candidate null beta0 (=log IRR0): impose beta_k=beta0 by shifting it into
the offset, refit restricted, draw the wild DGP from that null fit, refit
unrestricted on all 2^10=1024 sign vectors, and compute
p(beta0)=share(|t*(beta0)|>=|t_obs(beta0)|), with t=(beta_k - beta0)/se.
p(beta_hat_k)=1 (t_obs=0) and falls away from the point estimate; each CI
endpoint is the beta0 where p crosses 0.05, found by bisection walking out from
the peak. Point estimates unchanged. numpy+scipy+pandas only.
"""
import numpy as np, pandas as pd, itertools, os, sys
from numpy.linalg import solve, inv, pinv
from math import sqrt

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv("data/nymtc_safety_panel.csv").sort_values(
    ["County", "Year"]).reset_index(drop=True)
counties = sorted(df.County.unique()); years = sorted(df.Year.unique())

def design(d):
    c = {"const": np.ones(len(d))}
    for x in counties[1:]: c[f"cty_{x}"] = (d.County == x).astype(float)
    for y in years[1:]:    c[f"yr_{y}"]  = (d.Year == y).astype(float)
    return pd.DataFrame(c, index=d.index)

X = design(df); names = list(X.columns); Xv = X.values
off = np.log(df["VMT_100M"].values); groups = df.County.values
uniq = np.unique(groups); G = len(uniq); N, K = Xv.shape
year_cols = {nm: j for j, nm in enumerate(names) if nm.startswith("yr_")}
Gmat = np.array([(groups == g).astype(float) for g in uniq])
ss = (G / (G - 1)) * ((N - 1) / (N - K))
SIGNS = np.array(list(itertools.product([-1.0, 1.0], repeat=G))); Wobs = SIGNS @ Gmat
ALPHA = 0.05

def _inv(A):
    try: return inv(A)
    except np.linalg.LinAlgError: return pinv(A)

def qmle(Xm, y, off, b0, iters=60, tol=1e-11):
    b = b0.copy(); mu = np.exp(np.clip(Xm @ b + off, -30, 30)); sn = np.linalg.norm(Xm.T @ (y - mu))
    for _ in range(iters):
        A = Xm.T @ (Xm * mu[:, None]); g = Xm.T @ (y - mu)
        try: step = solve(A, g)
        except np.linalg.LinAlgError: step = pinv(A) @ g
        s = 1.0; ok = False
        for _ls in range(40):
            cand = b + s * step; mu_c = np.exp(np.clip(Xm @ cand + off, -30, 30)); sn_c = np.linalg.norm(Xm.T @ (y - mu_c))
            if np.all(np.isfinite(mu_c)) and sn_c <= sn + 1e-10: ok = True; break
            s *= 0.5
        if not ok: break
        b, mu, sn = cand, mu_c, sn_c
        if np.max(np.abs(s * step)) < tol: break
    return b, mu

def se_k_fun(Xm, y, mu, k):
    bread = _inv(Xm.T @ (Xm * mu[:, None]))
    S = Gmat @ (Xm * (y - mu)[:, None])
    cov = ss * (bread @ (S.T @ S) @ bread)
    v = cov[k, k]
    return sqrt(v) if v > 0 else np.inf

def p_at(beta0, k, b_hat, se_k, keep):
    off_b = off + beta0 * Xv[:, k]
    b_r_sub, mu_r = qmle(Xv[:, keep], y_g, off_b, b_hat[keep])
    resid = y_g - mu_r
    t_obs = (b_hat[k] - beta0) / se_k
    cnt = 0
    for bi in range(len(SIGNS)):
        ys = mu_r + Wobs[bi] * resid
        bb, mm = qmle(Xv, ys, off, b_hat)
        se_b = se_k_fun(Xv, ys, mm, k)
        if np.isfinite(se_b) and abs((bb[k] - beta0) / se_b) >= abs(t_obs): cnt += 1
    return cnt / len(SIGNS)

def cross(f, peak, far, target=ALPHA, it=15, tol=1e-3):
    """walk from peak (f=1>target) to far; return beta0 where f crosses target."""
    a, b = peak, far                                  # a:f>=target, b:f<target
    tries = 0
    while f(b) >= target and tries < 5:
        b = peak + 2.0 * (b - peak); tries += 1       # push out until below target
    for _ in range(it):
        if abs(b - a) < tol: break
        m = 0.5 * (a + b)
        if f(m) >= target: a = m
        else: b = m
    return 0.5 * (a + b)

rows = []
for label, col in [("Crashes", "crashes"), ("Fatalities", "fatalities"), ("Serious injuries", "serious_injuries")]:
    y_g = df[col].values.astype(float)
    b_hat, mu_hat = qmle(Xv, y_g, off, np.zeros(K))
    for nm, k in year_cols.items():
        yr = int(nm[3:]); se_k = se_k_fun(Xv, y_g, mu_hat, k); bk = b_hat[k]
        keep = [j for j in range(K) if j != k]
        f = lambda b0: p_at(b0, k, b_hat, se_k, keep)
        p0 = f(0.0)
        lo = cross(f, bk, bk - 6 * se_k)
        hi = cross(f, bk, bk + 6 * se_k)
        irr, ci_lo, ci_hi = float(np.exp(bk)), float(np.exp(lo)), float(np.exp(hi))
        cons = (ci_lo <= 1.0 <= ci_hi) == (p0 >= ALPHA)
        rows.append(dict(Outcome=label, Year=yr, IRR=round(irr, 4),
                         grid_CI_lo=round(ci_lo, 4), grid_CI_hi=round(ci_hi, 4),
                         wcr_p_at_IRR1=round(p0, 4),
                         contains_1=bool(ci_lo <= 1.0 <= ci_hi), sig_test=bool(p0 < ALPHA)))
        print(f"{label:16s} {yr}  IRR={irr:.3f}  grid-CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
              f"WCR_p(1)={p0:.4f}  1-in-CI={ci_lo<=1<=ci_hi}  sig={p0<ALPHA}  "
              f"{'ok' if cons else 'XX INCONSISTENT'}", flush=True)

allr = pd.DataFrame(rows)
allr.to_csv(f"{OUT}/r4_grid_inverted_ci.csv", index=False)
print(f"\nSaved: {OUT}/r4_grid_inverted_ci.csv")

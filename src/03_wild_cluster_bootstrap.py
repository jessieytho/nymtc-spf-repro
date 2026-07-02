"""
R4 - Few-cluster inference by wild cluster bootstrap-t for the SPF-offset model.

R1 estimator: Poisson QMLE, log(VMT) offset, county+year FE; overdispersion
(Pearson up to ~156) and within-county dependence carried by the cluster-robust
variance. With only G = 10 clusters the cluster-robust t over-rejects, so the
YEAR-effect inference is replaced by a wild cluster bootstrap-t (Cameron-Gelbach-
Miller), adapted to the count QMLE:

  * Wild DGP at cluster level on raw residuals: y*_i = mu_tilde_i + w_g (y_i - mu_tilde_i),
    Rademacher w_g in {-1,+1}.  mu_tilde is the RESTRICTED fit (null imposed,
    H0: IRR=1) for p-values [WCR], and the UNRESTRICTED fit for CIs [WCU-t].
  * Each wild sample is REFIT by QMLE and re-studentized, so beta* and se* both
    vary (the genuine bootstrap-t; avoids the w^2=1 degeneracy of a one-step
    score linearization, which spuriously narrows the interval).
  * Count QMLE solves X'(y* - mu)=0 directly with step-halving on the score norm,
    so it tolerates the (possibly negative) wild pseudo-values that a deviance-
    based Poisson IRLS cannot.
  * With G = 10, all 2^10 = 1,024 sign vectors are ENUMERATED EXACTLY -> no
    simulation error; min two-sided p = 1/1024 ~ 0.001.

Point estimates (IRRs) are UNCHANGED from R1; only the inference is refined.
Dependency-light: numpy + scipy + pandas only.
"""
import numpy as np, pandas as pd, itertools, os
from numpy.linalg import inv, pinv
from math import erf, sqrt

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("data/nymtc_safety_panel.csv").sort_values(
    ["County", "Year"]).reset_index(drop=True)
counties = sorted(df.County.unique()); years = sorted(df.Year.unique())

def design(d):
    cols = {"const": np.ones(len(d))}
    for c in counties[1:]: cols[f"cty_{c}"] = (d.County == c).astype(float)
    for y in years[1:]:    cols[f"yr_{y}"]  = (d.Year == y).astype(float)
    return pd.DataFrame(cols, index=d.index)

X = design(df); names = list(X.columns); Xv = X.values
off = np.log(df["VMT_100M"].values); groups = df.County.values
uniq_g = np.unique(groups); G = len(uniq_g); N, K = Xv.shape
year_cols = {nm: j for j, nm in enumerate(names) if nm.startswith("yr_")}
Gmat = np.array([(groups == g).astype(float) for g in uniq_g])
gidx = [np.where(groups == g)[0] for g in uniq_g]
ss = (G / (G - 1)) * ((N - 1) / (N - K))
SIGNS = np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
W_byobs = SIGNS @ Gmat

def qmle(Xm, y, off, iters=100, tol=1e-11):
    b = np.zeros(Xm.shape[1])
    pos = y[y > 0]
    b[0] = np.log(pos.sum() / np.exp(off).sum()) if pos.sum() > 0 else 0.0
    mu = np.exp(np.clip(Xm @ b + off, -30, 30))
    sn = np.linalg.norm(Xm.T @ (y - mu))
    for _ in range(iters):
        step = pinv(Xm.T @ (Xm * mu[:, None])) @ (Xm.T @ (y - mu))
        s = 1.0; ok = False
        for _ls in range(50):
            cand = b + s * step; mu_c = np.exp(np.clip(Xm @ cand + off, -30, 30))
            sn_c = np.linalg.norm(Xm.T @ (y - mu_c))
            if np.all(np.isfinite(mu_c)) and sn_c <= sn + 1e-10:
                ok = True; break
            s *= 0.5
        if not ok: break
        b, mu, sn = cand, mu_c, sn_c
        if np.max(np.abs(s * step)) < tol: break
    return b, mu

def se_cluster(Xm, y, mu, k):
    bread = inv(Xm.T @ (Xm * mu[:, None]))
    meat = np.zeros((Xm.shape[1],) * 2)
    for idx in gidx:
        sg = Xm[idx].T @ (y[idx] - mu[idx]); meat += np.outer(sg, sg)
    cov = ss * (bread @ meat @ bread)
    return sqrt(cov[k, k])

def ncdf(z): return 0.5 * (1 + erf(z / sqrt(2)))

def infer(label, col):
    y = df[col].values.astype(float)
    b_hat, mu_hat = qmle(Xv, y, off)
    pearson = np.sum((y - mu_hat) ** 2 / mu_hat) / (N - K)

    resid_u = y - mu_hat
    bstar = np.full((len(SIGNS), K), np.nan); sestar = np.full((len(SIGNS), K), np.nan)
    for bi in range(len(SIGNS)):
        ys = mu_hat + W_byobs[bi] * resid_u
        bb, mm = qmle(Xv, ys, off)
        bstar[bi] = bb
        for nm, k in year_cols.items():
            sestar[bi, k] = se_cluster(Xv, ys, mm, k)

    rows = []
    for nm, k in year_cols.items():
        yr = int(nm[3:]); irr = float(np.exp(b_hat[k]))
        se_k = se_cluster(Xv, y, mu_hat, k); t_obs = b_hat[k] / se_k
        asym_lo, asym_hi = np.exp(b_hat[k] - 1.96*se_k), np.exp(b_hat[k] + 1.96*se_k)
        p_asym = 2 * (1 - ncdf(abs(t_obs)))

        tci = (bstar[:, k] - b_hat[k]) / sestar[:, k]
        qlo, qhi = np.quantile(tci, [0.025, 0.975])
        ci_lo = float(np.exp(b_hat[k] - qhi * se_k))
        ci_hi = float(np.exp(b_hat[k] - qlo * se_k))
        crit = float(np.quantile(np.abs(tci), 0.95))

        keep = [j for j in range(K) if j != k]
        b_r_sub, _ = qmle(Xv[:, keep], y, off)
        b_r = np.zeros(K); b_r[keep] = b_r_sub
        mu_r = np.exp(np.clip(Xv @ b_r + off, -30, 30)); resid_r = y - mu_r
        cnt = 0
        for bi in range(len(SIGNS)):
            ys = mu_r + W_byobs[bi] * resid_r
            bb, mm = qmle(Xv, ys, off)
            se_b = se_cluster(Xv, ys, mm, k)
            if abs(bb[k] / se_b) >= abs(t_obs): cnt += 1
        p_wcr = cnt / len(SIGNS)

        rows.append(dict(Outcome=label, Year=yr, IRR=round(irr, 4),
                         asym_CI_lo=round(float(asym_lo), 4), asym_CI_hi=round(float(asym_hi), 4),
                         asym_p=round(float(p_asym), 4),
                         wcb_CI_lo=round(ci_lo, 4), wcb_CI_hi=round(ci_hi, 4),
                         wcb_t_crit95=round(crit, 3), wcr_p=round(p_wcr, 4),
                         sig_asym=bool(p_asym < .05), sig_wcb=bool(p_wcr < .05)))
    return pearson, pd.DataFrame(rows)

print("=" * 90)
print("R4  WILD CLUSTER BOOTSTRAP-t  (refit, Rademacher, exact 2^10=1024; point est. unchanged)")
print("=" * 90)
res = []
for label, col in [("Crashes", "crashes"), ("Fatalities", "fatalities"),
                   ("Serious injuries", "serious_injuries")]:
    pe, t = infer(label, col); res.append(t)
    print(f"\n### {label}   (Poisson Pearson dispersion = {pe:.2f})")
    print("  Yr     IRR   asym[lo,hi] asym_p   WCB[lo,hi]  t*crit  WCR_p   sig asym->WCB")
    for _, r in t.iterrows():
        flip = "" if r.sig_asym == r.sig_wcb else "   <== CHANGES"
        print(f"  {r.Year}  {r.IRR:5.3f}  [{r.asym_CI_lo:.2f},{r.asym_CI_hi:.2f}] {r.asym_p:6.3f}"
              f"  [{r.wcb_CI_lo:.2f},{r.wcb_CI_hi:.2f}] {r.wcb_t_crit95:5.2f}  {r.wcr_p:6.3f}"
              f"  {str(r.sig_asym):>5}->{str(r.sig_wcb):>5}{flip}")

allr = pd.concat(res, ignore_index=True)
allr.to_csv(f"{OUT}/r4_wildbootstrap_inference.csv", index=False)
print(f"\nSaved: {OUT}/r4_wildbootstrap_inference.csv")
print(f"[note] min achievable two-sided WCR p with G={G}: 1/2^{G} = {1/2**G:.4f}")

"""Experiment B (coverage / Type-I): SPF-offset year-effect inference, comparing a
Poisson model-based SE, a cluster-robust sandwich, and the wild cluster bootstrap,
across cluster count and dispersion. Default run is reduced for speed; the paper used
R=220, B=149. Writes outputs/v2_coverage_table.csv."""
import numpy as np, os, csv, time
from numpy.linalg import solve, pinv, inv
from math import erf, sqrt
RNG=np.random.default_rng(7); T=6; RHO=2.3e-3; R=int(os.getenv("COV_R","140")); B=int(os.getenv("COV_B","79"))
OUT="outputs"; os.makedirs(OUT,exist_ok=True); CSV=f"{OUT}/v2_coverage_table.csv"
def make_exposure(G): return np.exp(np.random.default_rng(100+G).normal(np.log(8000),1.0,G))
def design(G,tested=T-1):
    cid=np.repeat(np.arange(G),T); yr=np.tile(np.arange(T),G)
    cols=[np.ones(G*T)]+[(cid==c).astype(float) for c in range(1,G)]+[(yr==t).astype(float) for t in range(1,T)]
    return np.column_stack(cols),1+(G-1)+[t for t in range(1,T)].index(tested),cid
def qmle(X,y,off,b0=None,it=40,tol=1e-9):
    b=np.zeros(X.shape[1]) if b0 is None else b0.copy()
    if b0 is None:
        pos=y[y>0]; b[0]=np.log(pos.sum()/np.exp(off).sum()) if pos.sum()>0 else 0.0
    mu=np.exp(np.clip(X@b+off,-30,30)); sn=np.linalg.norm(X.T@(y-mu))
    for _ in range(it):
        A=X.T@(X*mu[:,None]); g=X.T@(y-mu)
        try: step=solve(A,g)
        except: step=pinv(A)@g
        s=1.0; ok=False
        for _ls in range(30):
            cand=b+s*step; mc=np.exp(np.clip(X@cand+off,-30,30)); snc=np.linalg.norm(X.T@(y-mc))
            if np.all(np.isfinite(mc)) and snc<=sn+1e-10: ok=True; break
            s*=0.5
        if not ok: break
        b,mu,sn=cand,mc,snc
        if np.max(np.abs(s*step))<tol: break
    return b,mu
def se_model(X,mu,k): return sqrt(inv(X.T@(X*mu[:,None]))[k,k])
def se_k(X,y,mu,k,G,cid):
    bread=inv(X.T@(X*mu[:,None])); K=X.shape[1]; N=X.shape[0]; meat=np.zeros((K,K))
    for c in range(G):
        idx=cid==c; sg=X[idx].T@(y[idx]-mu[idx]); meat+=np.outer(sg,sg)
    corr=(G/(G-1))*((N-1)/(N-K)); v=(corr*(bread@meat@bread))[k,k]; return sqrt(v) if v>0 else np.inf
def ncdf(z): return 0.5*(1+erf(z/sqrt(2)))
def wcr_p(X,k,y,off,G,cid,logh,b_hat,se_hat):
    t_obs=(b_hat[k]-logh)/se_hat; keep=[j for j in range(X.shape[1]) if j!=k]; offr=off+logh*X[:,k]
    br,mur=qmle(X[:,keep],y,offr); resid=y-mur; cnt=0
    for _ in range(B):
        ys=mur+RNG.choice([-1.0,1.0],G)[cid]*resid; bb,mm=qmle(X,ys,off,b0=b_hat); seb=se_k(X,ys,mm,k,G,cid)
        if np.isfinite(seb) and abs((bb[k]-logh)/seb)>=abs(t_obs): cnt+=1
    return cnt/B
def run_cell(G,disp,g_true,test_at,label):
    X,k,cid=design(G); E=make_exposure(G); off=np.log(np.repeat(E,T)); yr=np.tile(np.arange(T),G)
    ri=rc=rw=n=0; logh=np.log(test_at)
    for _ in range(R):
        ye=np.ones(T); ye[T-1]=g_true; mean=np.repeat(E,T)*RHO*ye[yr]
        y=(RNG.poisson(mean) if disp=="pois" else RNG.negative_binomial(1/float(disp),(1/float(disp))/((1/float(disp))+mean))).astype(float)
        b_hat,mu=qmle(X,y,off); s=se_k(X,y,mu,k,G,cid)
        if not np.isfinite(s) or s<=0: continue
        n+=1
        if 2*(1-ncdf(abs((b_hat[k]-logh)/se_model(X,mu,k))))<0.05: ri+=1
        if 2*(1-ncdf(abs((b_hat[k]-logh)/s)))<0.05: rc+=1
        if wcr_p(X,k,y,off,G,cid,logh,b_hat,s)<0.05: rw+=1
    return dict(cell=label,G=G,disp=disp,g_true=g_true,test_at=test_at,n=n,
                IMnormal_rej=round(ri/n,3),CRnormal_rej=round(rc/n,3),WCR_rej=round(rw/n,3))
cells=[(10,"pois",1.0,1.0,"TypeI_G10_Pois"),(10,"0.5",1.0,1.0,"TypeI_G10_NB"),
       (6,"0.5",1.0,1.0,"TypeI_G6_NB"),(20,"0.5",1.0,1.0,"TypeI_G20_NB"),
       (10,"pois",1.3,1.3,"Coverage_at_1.3_G10")]
rows=[run_cell(*c) for c in cells]
with open(CSV,"w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); [wr.writerow(r) for r in rows]
for r in rows: print(r)
print("Saved",CSV,"(R=%d,B=%d)"%(R,B))
# --- Figure (Experiment B): Type-I rejection by procedure / cell ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
order=["TypeI_G10_Pois","TypeI_G6_NB","TypeI_G10_NB","TypeI_G20_NB"]
labels=["G=10\nPoisson\n(model correct)","G=6\nNB overdisp.","G=10\nNB overdisp.","G=20\nNB overdisp."]
d={r["cell"]:r for r in rows}
im=[d[c]["IMnormal_rej"] for c in order]; cr=[d[c]["CRnormal_rej"] for c in order]; wc=[d[c]["WCR_rej"] for c in order]
x=np.arange(len(order)); w=0.26
fig,ax=plt.subplots(figsize=(8,4.5))
bsets=[ax.bar(x-w,im,w,label="Poisson model-based SE (IM)",color="#b2182b"),
       ax.bar(x,cr,w,label="Cluster-robust sandwich (normal)",color="#f4a582"),
       ax.bar(x+w,wc,w,label="Wild cluster bootstrap (WCR)",color="#2166ac")]
ax.axhline(0.05,ls="--",lw=1,color="k"); ax.text(len(order)-0.55,0.065,"nominal 5%",fontsize=8)
for bars in bsets:
    for bb in bars: ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+0.008,f"{bb.get_height():.2f}",ha="center",fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8)
ax.set_ylabel("Type-I rejection of a true null (nominal 0.05)")
ax.set_title("Model-based SEs collapse under overdispersion; robust SEs over-reject at few\nclusters; the wild cluster bootstrap stays best-calibrated",fontsize=9)
ax.legend(fontsize=8,loc="upper right"); ax.set_ylim(0,0.85)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_v2_coverage.png",dpi=200,bbox_inches="tight")
print("Saved",f"{OUT}/fig_v2_coverage.png")


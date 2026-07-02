"""Encodes the paper's central simulation claim as a test: under a rare-event,
smooth-denominator DGP with a KNOWN true ratio, the conditional estimator is
~unbiased and tight, while the count-on-count slope ratio is unidentified
(median ~unbiased but wide, with sign flips)."""
import numpy as np
from numpy.linalg import pinv
def run(R=1500,seed=0):
    rng=np.random.default_rng(seed); G,T=10,6; N=G*T
    cid=np.repeat(np.arange(G),T); yr=np.tile(np.arange(T),G)
    Z=np.column_stack([np.ones(N)]+[(cid==c).astype(float) for c in range(1,G)]
                      +[(yr==t).astype(float) for t in range(1,T)])
    M=np.eye(N)-Z@pinv(Z)
    base=np.linspace(2000,30000,G); rB=2.2e-3; tau=1.91; lamA,lamB=0.15,0.05
    def fes(C,F): Cx=C@M; return np.sum(Cx*F,1)/np.sum(Cx*Cx,1)
    Ct=np.maximum(rng.poisson(0.046*base[cid]*np.exp(lamA*rng.standard_normal((R,N)))),1).astype(float)
    Cn=np.maximum(rng.poisson(base[cid]*np.exp(lamB*rng.standard_normal((R,N)))),1).astype(float)
    Ft=rng.poisson(rB*tau*Ct).astype(float); Fn=rng.poisson(rB*Cn).astype(float)
    wrong=fes(Ct,Ft)/fes(Cn,Fn)
    right=(Ft.sum(1)/Ct.sum(1))/(Fn.sum(1)/Cn.sum(1))
    return wrong,right,tau
def test_instability():
    w,r,tau=run()
    wr=lambda a:np.nanpercentile(a,97.5)-np.nanpercentile(a,2.5)
    assert abs(np.nanmedian(r)-tau)<0.15, "conditional estimator should be ~unbiased"
    assert wr(r)<0.8, "conditional estimator should be tight"
    assert wr(w)>3*wr(r), "count-on-count should be far wider (unidentified)"
    assert np.mean(w<0)>0.0, "count-on-count should sometimes flip sign"
if __name__=="__main__":
    test_instability(); print("PASS test_simulation_instability")

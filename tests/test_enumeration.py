"""Property test: the exact wild-bootstrap sign-vector enumeration for G clusters
yields all 2^G Rademacher vectors, unique, in {-1,+1}, and closed under negation."""
import numpy as np, itertools
def enumerate_signs(G): return np.array(list(itertools.product([-1.0,1.0],repeat=G)))
def test_enumeration_G10():
    G=10; S=enumerate_signs(G)
    assert S.shape==(1024,G), f"expected 1024x{G}, got {S.shape}"
    assert set(np.unique(S))=={-1.0,1.0}, "entries must be +/-1"
    assert len({tuple(r) for r in S})==1024, "vectors must be unique"
    # closed under negation (symmetry of the Rademacher bootstrap)
    sset={tuple(r) for r in S}
    assert all(tuple(-r) in sset for r in S), "set not closed under negation"
def test_counts_small_G():
    for G in (1,4,6,8): assert enumerate_signs(G).shape[0]==2**G
if __name__=="__main__":
    test_enumeration_G10(); test_counts_small_G(); print("PASS test_enumeration")

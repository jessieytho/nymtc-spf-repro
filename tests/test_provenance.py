"""Exposure-provenance assertion. The provenance script (src/01) runs in one of
two modes (see its docstring): LIVE when the user-supplied NYSDOT DVMT workbook is
present, or DOCUMENTED when it is absent (the shipped default, since that workbook
is third-party and not redistributed). This test passes in either mode: it requires
the script to exit 0 and to report the ratio = 1.0000 provenance identity (asserted
live, or recorded and re-verifiable in documented mode)."""
import subprocess, sys, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_provenance_ratio():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "01_verify_vmt_provenance.py")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"provenance script failed:\n{r.stderr[-500:]}"
    out = r.stdout + r.stderr
    assert ("[LIVE]" in out) or ("[DOCUMENTED]" in out), \
        f"provenance script reported neither LIVE nor DOCUMENTED mode:\n{out[-600:]}"
    nums = [float(x) for x in re.findall(r"1\.000\d?|0\.999\d|1\.001\d|ratio[^0-9]*([0-9.]+)", out) if x]
    assert ("1.0000" in out) or any(abs(n - 1) < 1e-3 for n in nums), \
        f"provenance ratio not ~1.0000 in output:\n{out[-600:]}"


if __name__ == "__main__":
    test_provenance_ratio(); print("PASS test_provenance")

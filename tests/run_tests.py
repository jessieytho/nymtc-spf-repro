"""Run all repository tests without pytest. Exits non-zero on first failure."""
import importlib.util, glob, os, sys, traceback
here=os.path.dirname(os.path.abspath(__file__)); fails=0
for f in sorted(glob.glob(os.path.join(here,"test_*.py"))):
    spec=importlib.util.spec_from_file_location(os.path.basename(f)[:-3],f)
    mod=importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        for name in [n for n in dir(mod) if n.startswith("test_")]:
            getattr(mod,name)()
        print(f"PASS  {os.path.basename(f)}")
    except Exception:
        fails+=1; print(f"FAIL  {os.path.basename(f)}"); traceback.print_exc()
print(f"\n{'ALL TESTS PASSED' if not fails else str(fails)+' TEST FILE(S) FAILED'}")
sys.exit(1 if fails else 0)

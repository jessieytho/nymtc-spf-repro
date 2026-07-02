"""Regenerate every table and figure from source, in order. Run from repo root:
    python run_all.py
Reads ./data, writes ./outputs. Dependency-light (see requirements.txt)."""
import subprocess, sys, os, glob
os.makedirs("outputs", exist_ok=True)
steps = sorted(glob.glob("src/[0-9][0-9]_*.py"))
for s in steps:
    print("\n" + "="*70 + f"\nRUN  {s}\n" + "="*70)
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
        print(f"FAILED at {s}"); sys.exit(r.returncode)
print("\nAll steps complete. See ./outputs.")
print("Run the test suite with:  python tests/run_tests.py")

import shutil
os.makedirs("figures", exist_ok=True)
for _png in glob.glob("outputs/*.png"):
    shutil.copy(_png, os.path.join("figures", os.path.basename(_png)))
print("Mirrored figures/*.png for LaTeX (\\graphicspath includes figures/).")

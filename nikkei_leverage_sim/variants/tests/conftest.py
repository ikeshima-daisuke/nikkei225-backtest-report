"""Make the core package (src/) and the variants package importable.

Kept local to the variants test tree so the core ``pytest`` configuration
(testpaths=["tests"], pythonpath=["src"]) is untouched.
"""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]      # nikkei_leverage_sim/
SRC = PKG / "src"
for p in (str(SRC), str(PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

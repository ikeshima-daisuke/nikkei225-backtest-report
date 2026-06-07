"""Make the core package (src/) and sibling sub-packages importable.

Mirrors ``accumulation_study/tests/conftest.py`` so the core pytest config
(testpaths=["tests"], pythonpath=["src"]) stays untouched and this tree runs
standalone via ``pytest regime_study/tests``.
"""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]  # nikkei_leverage_sim/
SRC = PKG / "src"
for p in (str(SRC), str(PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

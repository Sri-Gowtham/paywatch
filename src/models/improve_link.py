"""Phase 1 candidate xgb_link (E2): xgb_v1 configuration plus 20 causal linkage features."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(sorted(glob.glob("/kaggle/input/**/common_improve.py", recursive=True))[0]))
import common_improve as ci
D = ci.load(linkage=True)
ci.run_candidate("xgb_link", ci.xgb_family(ci.XGB_BASE, patience=300), D)

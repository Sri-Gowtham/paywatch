"""Phase 1 candidate xgb_v2: depth 12 (best validation depth in Phase 0), patience 300."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(sorted(glob.glob("/kaggle/input/**/common_improve.py", recursive=True))[0]))
import common_improve as ci
D = ci.load()
ci.run_candidate("xgb_v2", ci.xgb_family({**ci.XGB_BASE, "max_depth": 12}, patience=300), D)

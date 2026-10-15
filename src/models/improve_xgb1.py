"""Phase 1 candidate xgb_v1: production XGBoost config (depth 10) with early-stopping patience 300 instead of 60."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(sorted(glob.glob("/kaggle/input/**/common_improve.py", recursive=True))[0]))
import common_improve as ci
D = ci.load()
ci.run_candidate("xgb_v1", ci.xgb_family(ci.XGB_BASE, patience=300), D)

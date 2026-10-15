"""Phase 1 candidate lgbm: LightGBM (E1 ensemble component)."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(sorted(glob.glob("/kaggle/input/**/common_improve.py", recursive=True))[0]))
import common_improve as ci
D = ci.load()
ci.run_candidate("lgbm", ci.lgbm_family(patience=300), D)

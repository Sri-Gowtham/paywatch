"""Phase 1 candidate cat: CatBoost (E1 ensemble component); 2 seeds to bound runtime."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(sorted(glob.glob("/kaggle/input/**/common_improve.py", recursive=True))[0]))
import common_improve as ci
D = ci.load()
ci.run_candidate("cat", ci.catboost_family(patience=200), D, seeds=(42, 1))

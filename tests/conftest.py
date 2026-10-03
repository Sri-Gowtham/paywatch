import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_DIR = os.environ.get("PAYWATCH_FEATURES_DIR", os.path.join(ROOT, "src", "features"))
for p in (ROOT, FEATURES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

MODELS_DIR = os.environ.get("PAYWATCH_MODELS_DIR", os.path.join(ROOT, "models", "compact"))
FIXTURE = os.environ.get("PAYWATCH_FIXTURE", os.path.join(MODELS_DIR, "parity_fixture.json"))

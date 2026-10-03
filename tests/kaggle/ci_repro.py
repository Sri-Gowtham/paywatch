"""Reproduce the GitHub Actions 'test' job on Kaggle (internet ON): clone the public repo, install
requirements-dev.txt (latest versions, like CI) into a private target dir, compileall, pytest -q tests.
A venv is not used: Kaggle's sitecustomize breaks ensurepip inside venvs."""

import json
import os
import subprocess
import sys

REPO = "https://github.com/Sri-Gowtham/paywatch"
WORK = "/tmp/ci"
TARGET = f"{WORK}/site"
REF = os.environ.get("CI_REF", "main")
results = {}


def run(cmd, cwd=None, name=None, pythonpath=None):
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    print(f"$ {' '.join(cmd)}\n-> rc={p.returncode}\n{out[-6000:]}\n", flush=True)
    if name:
        results[name] = {"rc": p.returncode, "tail": out[-12000:]}
    return p


os.makedirs(WORK, exist_ok=True)
run(["git", "clone", "--depth", "1", "--branch", REF, REPO, f"{WORK}/repo"], name="clone")
results["commit"] = run(["git", "rev-parse", "HEAD"], cwd=f"{WORK}/repo").stdout.strip()
import glob
import shutil

if os.environ.get("OVERLAY", "1") == "1":
    # test the CURRENT working tree: copy the local files (uploaded as Kaggle datasets) over the clone
    find = lambda pat: sorted(glob.glob(pat, recursive=True))  # noqa: E731
    for marker, dest in (("upi_fingerprint.py", "src/features"), ("snapshot.py", "src/api"),
                         ("drift_detector.py", "src/monitor"), ("qa_chain.py", "src/rag"), ("test_api.py", "tests")):
        hits = find(f"/kaggle/input/**/{marker}")
        if hits:
            src = os.path.dirname(hits[0])
            os.makedirs(f"{WORK}/repo/{dest}", exist_ok=True)
            for f in os.listdir(src):
                if f.endswith(".py"):
                    shutil.copy(os.path.join(src, f), f"{WORK}/repo/{dest}/{f}")
            print(f"overlaid {dest} from {src}", flush=True)
    results["overlay"] = True

run([sys.executable, "-m", "pip", "install", "-q", "--target", TARGET, "-r", "requirements-dev.txt"],
    cwd=f"{WORK}/repo", name="pip_install")
vers = run([sys.executable, "-c",
            "import importlib.metadata as m;print([n+'=='+m.version(n) for n in "
            "['fastapi','pydantic','numpy','pandas','xgboost','pytest','httpx','uvicorn','starlette']])"],
           pythonpath=TARGET).stdout
results["versions"] = vers.strip()
results["python"] = sys.version.split()[0]
run([sys.executable, "-m", "compileall", "-q", "src"], cwd=f"{WORK}/repo", name="compileall", pythonpath=TARGET)
run([sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", "tests"], cwd=f"{WORK}/repo",
    name="pytest", pythonpath=TARGET)
# ---------------------------------------------------------------- slim runtime check (stand-in for the Docker image)
# Only requirements-api.txt is installed and `python -S` hides every other installed package, so anything the API
# imports beyond numpy/xgboost/fastapi/uvicorn/pydantic fails here exactly as it would in the container.
import time
import urllib.request

API_TARGET = f"{WORK}/site_api"
run([sys.executable, "-m", "pip", "install", "-q", "--target", API_TARGET, "-r", "requirements-api.txt"],
    cwd=f"{WORK}/repo", name="pip_install_api_only")
slim = {"pandas_importable_in_slim_env": None, "health": None, "predict": None}
probe = subprocess.run([sys.executable, "-S", "-c", "import pandas"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": API_TARGET})
slim["pandas_importable_in_slim_env"] = probe.returncode == 0
env = {**os.environ, "PYTHONPATH": f"{API_TARGET}:{WORK}/repo", "PAYWATCH_MODELS_DIR": f"{WORK}/repo/models/compact"}
server = subprocess.Popen([sys.executable, "-S", "-m", "uvicorn", "--factory", "src.api.main:create_app",
                           "--host", "127.0.0.1", "--port", "8766", "--workers", "1"], cwd=f"{WORK}/repo", env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    for _ in range(60):
        try:
            slim["health"] = json.loads(urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=2).read())
            break
        except Exception:
            time.sleep(1)
    if slim["health"]:
        row = json.load(open(f"{WORK}/repo/models/compact/parity_fixture.json"))["rows"][0]
        req = urllib.request.Request("http://127.0.0.1:8766/predict", method="POST",
                                     data=json.dumps({"transaction_id": "slim-1", "fields": row["fields"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        body = json.loads(urllib.request.urlopen(req, timeout=20).read())
        slim["predict"] = {k: body[k] for k in ("action", "score", "calibrated_probability", "model_version")}
        slim["predict"]["n_reasons"] = len(body["reasons"])
finally:
    server.terminate()
    slim["server_log_tail"] = (server.communicate(timeout=10)[0] or "")[-600:]
results["slim_runtime"] = slim
print("slim runtime:", json.dumps({k: v for k, v in slim.items() if k != "server_log_tail"}), flush=True)
json.dump(results, open("/kaggle/working/results.json", "w"), indent=1)
print("\nSUMMARY:", {k: v["rc"] for k, v in results.items() if isinstance(v, dict) and "rc" in v})
print("versions:", results["versions"])

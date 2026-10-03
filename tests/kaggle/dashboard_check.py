"""PayWatch — dashboard test (Kaggle, CPU, internet ON only for pip).

Starts the real API (uvicorn in a thread) and drives the Streamlit dashboard with streamlit.testing AppTest:
renders all tabs, moves the threshold slider, switches scenario, and clicks 'Score transaction' against the API.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "streamlit", "fastapi", "uvicorn"], check=True)

find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
REPO, MODELS, OUT = "/kaggle/working/repo", "/kaggle/working/models/compact", "/kaggle/working"
for d in (f"{REPO}/src/api", f"{REPO}/src/monitor", f"{REPO}/src/rag", f"{REPO}/data/sim", f"{REPO}/data/rag", MODELS):
    os.makedirs(d, exist_ok=True)
for marker, dest in (("snapshot.py", "api"), ("dashboard.py", "monitor"), ("qa_chain.py", "rag")):
    p = find(f"/kaggle/input/**/{marker}")[0]
    for f in os.listdir(os.path.dirname(p)):
        if f.endswith(".py"):
            shutil.copy(os.path.join(os.path.dirname(p), f), f"{REPO}/src/{dest}/{f}")
open(f"{REPO}/src/__init__.py", "w").close()
models_src = os.path.dirname(find("/kaggle/input/**/xgb_seed42.json")[0])
for f in os.listdir(models_src):
    shutil.copy(os.path.join(models_src, f), MODELS)
shutil.copy(find("/kaggle/input/**/sim_results.json")[0], f"{REPO}/data/sim/sim_results.json")
shutil.copy(find("/kaggle/input/**/chunks.jsonl")[0], f"{REPO}/data/rag/chunks.jsonl")
sys.path.insert(0, REPO)
os.environ.update(PAYWATCH_MODELS_DIR=MODELS, PAYWATCH_SIM_RESULTS=f"{REPO}/data/sim/sim_results.json",
                  PAYWATCH_API_URL="http://127.0.0.1:8765")

import uvicorn  # noqa: E402
from src.api.main import create_app  # noqa: E402

server = uvicorn.Server(uvicorn.Config(create_app(MODELS), host="127.0.0.1", port=8765, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=1)
        break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("API did not start")
print("API up")

from streamlit.testing.v1 import AppTest  # noqa: E402

checks = {}


def check(name, cond, detail=""):
    checks[name] = {"pass": bool(cond), "detail": str(detail)}
    print(("PASS " if cond else "FAIL ") + name, detail if not cond else "")


at = AppTest.from_file(f"{REPO}/src/monitor/dashboard.py", default_timeout=120).run()
check("renders without exception", not at.exception, [e.value for e in at.exception])
check("five tabs", len(at.tabs) == 5, len(at.tabs))
check("title present", any("PayWatch" in t.value for t in at.title))

# threshold slider
slider = next(s for s in at.slider if "threshold" in s.label.lower())
lo = [m.value for m in at.metric if m.label == "Precision"][0]
at = slider.set_value(0.60).run()
hi = [m.value for m in at.metric if m.label == "Precision"][0]
check("threshold slider changes precision", lo != hi, f"{lo} -> {hi}")
check("no exception after slider", not at.exception, [e.value for e in at.exception])

# scenario radio
radio = next(r for r in at.radio if r.label == "Scenario")
at = radio.set_value("control").run()
check("control scenario renders", not at.exception, [e.value for e in at.exception])
control_alerts = {m.label: m.value for m in at.metric if "alerts" in m.label.lower()}
at = next(r for r in at.radio if r.label == "Scenario").set_value("drift").run()
drift_alerts = {m.label: m.value for m in at.metric if "alerts" in m.label.lower()}
print("control:", control_alerts, "| drift:", drift_alerts)

# cost inputs change the cheapest threshold text
at = next(n for n in at.number_input if n.label.startswith("Cost of a missed")).set_value(100000.0).run()
check("cost input rerun ok", not at.exception, [e.value for e in at.exception])

# live scoring against the running API
at = next(b for b in at.button if b.label == "Score transaction").click().run()
check("live scoring shows metrics", any(m.label == "Calibrated probability" for m in at.metric) and not at.error,
      [e.value for e in at.error])
if any(m.label == "Action" for m in at.metric):
    print("live scoring action:", [m.value for m in at.metric if m.label == "Action"])

# analyst assistant: answers from the RBI documents, refuses off-topic, explains the scored transaction
all_md = lambda a: " ".join(m.value for m in a.markdown)  # noqa: E731
check("assistant answers with a citation", any("[1]" in m.value for m in at.markdown) or "working days" in all_md(at).lower(),
      all_md(at)[:200])
rag_q = next(t for t in at.text_input if t.label.startswith("Ask about"))
at_off = rag_q.set_value("How do I cook biryani?").run()
check("assistant refuses off-topic question", "not covered" in all_md(at_off), all_md(at_off)[:200])
check("explanation of the last scored transaction", any("This transaction was" in t.value for t in at.text), [t.value for t in at.text][:2])
check("explanation shows RBI context", "Relevant RBI passages" in all_md(at), "")

# API down -> graceful error
os.environ["PAYWATCH_API_URL"] = "http://127.0.0.1:9"
at2 = AppTest.from_file(f"{REPO}/src/monitor/dashboard.py", default_timeout=120).run()
at2.text_input[0].set_value("http://127.0.0.1:9").run()
at2 = next(b for b in at2.button if b.label == "Score transaction").click().run()
check("API down handled gracefully", bool(at2.error) and not at2.exception, [e.value for e in at2.exception])

results = {"checks": checks, "all_pass": all(c["pass"] for c in checks.values()),
           "alert_metrics": {"control": control_alerts, "drift": drift_alerts}}
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
shutil.rmtree(REPO, ignore_errors=True)
print("\nALL PASS" if results["all_pass"] else "\nSOME CHECKS FAILED")
server.should_exit = True

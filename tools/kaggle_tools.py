"""Kaggle workflow helper: all heavy compute (features, training, evaluation, simulation, RAG build) runs on Kaggle.

  python tools/kaggle_tools.py list
  python tools/kaggle_tools.py push <kernel> [--wait]          stage script + metadata from kaggle/kernels.json, push
  python tools/kaggle_tools.py status [<kernel> ...]
  python tools/kaggle_tools.py fetch <kernel> [--pattern results.json] [--out results/<kernel>.json]
  python tools/kaggle_tools.py version <dataset> -m "message"   dataset keys come from kaggle/datasets.json

`fetch` downloads only files matching --pattern (never whole outputs: some kernels write large model files).
Credentials: ~/.kaggle/access_token (never committed).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNELS = json.load(open(os.path.join(ROOT, "kaggle", "kernels.json"), encoding="utf-8"))
DATASETS = json.load(open(os.path.join(ROOT, "kaggle", "datasets.json"), encoding="utf-8"))
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def kaggle(*args, timeout=300, cwd=None):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, env=ENV, timeout=timeout, cwd=cwd)


def kernel_id(name):
    if name not in KERNELS:
        sys.exit(f"unknown kernel '{name}'. Known: {', '.join(KERNELS)}")
    return KERNELS[name]["metadata"]["id"]


def status(name):
    out = (kaggle("kernels", "status", kernel_id(name), timeout=60).stdout or "").strip()
    return out.split("status ")[-1].strip('"') if out else "UNKNOWN"


def push(name, wait=False, max_wait_s=7200):
    if name not in KERNELS:
        sys.exit(f"unknown kernel '{name}'. Known: {', '.join(KERNELS)}")
    spec = KERNELS[name]
    script = os.path.join(ROOT, spec["script"])
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(script, os.path.join(tmp, spec["metadata"]["code_file"]))
        json.dump(spec["metadata"], open(os.path.join(tmp, "kernel-metadata.json"), "w"), indent=2)
        r = kaggle("kernels", "push", "-p", tmp, timeout=120)
    print((r.stdout or r.stderr).strip())
    if r.returncode or not wait:
        return r.returncode
    time.sleep(60)                                    # status can still show the previous run for a moment
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        s = status(name)
        if "COMPLETE" in s or "ERROR" in s or "CANCEL" in s:
            print(name, "->", s)
            return 0 if "COMPLETE" in s else 1
        time.sleep(20)
    print(f"{name}: still running after {max_wait_s}s; stopped waiting (the kernel keeps running on Kaggle)")
    return 2


def fetch(name, pattern, out):
    out = out or os.path.join(ROOT, "results", f"{name}.json")
    with tempfile.TemporaryDirectory() as tmp:
        r = kaggle("kernels", "output", kernel_id(name), "-p", tmp, "--file-pattern", pattern, timeout=150)
        files = [f for f in os.listdir(tmp) if not f.endswith(".log")]
        if r.returncode or not files:
            print(f"{name}: nothing fetched ({(r.stderr or r.stdout).strip()[:200]})")
            return 1
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy(os.path.join(tmp, files[0]), out)
    try:
        shown = os.path.relpath(out, ROOT)
    except ValueError:                                # --out on another drive (Windows)
        shown = out
    print(f"{name}: {files[0]} -> {shown} ({os.path.getsize(out):,} bytes)")
    return 0


def version(key, message):
    if key not in DATASETS:
        sys.exit(f"unknown dataset '{key}'. Known: {', '.join(DATASETS)}")
    # run from inside the folder: the CLI builds upload-cache file names from the -p path and breaks on 'src/models'
    r = kaggle("datasets", "version", "-p", ".", "-m", message, timeout=600, cwd=os.path.join(ROOT, DATASETS[key]["path"]))
    text = ((r.stdout or "") + (r.stderr or "")).strip()
    ok = "being created" in text
    print(f"{key}: " + ("version created" if ok else "FAILED: " + "; ".join(text.splitlines()[-3:])))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("push")
    p.add_argument("kernel")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--max-wait", type=int, default=7200, help="seconds to wait with --wait (default 7200)")
    p = sub.add_parser("status")
    p.add_argument("kernels", nargs="*")
    p = sub.add_parser("fetch")
    p.add_argument("kernel")
    p.add_argument("--pattern", default="results.json")
    p.add_argument("--out")
    p = sub.add_parser("version")
    p.add_argument("dataset")
    p.add_argument("-m", "--message", required=True)
    a = ap.parse_args()
    if a.cmd == "list":
        for k, v in KERNELS.items():
            print(f"{k:<18} {v['metadata']['id']:<36} {v['script']}")
        print()
        for k, v in DATASETS.items():
            print(f"dataset {k:<12} {v['id']:<36} {v['path']}")
    elif a.cmd == "push":
        sys.exit(push(a.kernel, a.wait, a.max_wait))
    elif a.cmd == "status":
        for k in a.kernels or KERNELS:
            print(f"{k:<18} {status(k)}")
    elif a.cmd == "fetch":
        sys.exit(fetch(a.kernel, a.pattern, a.out))
    elif a.cmd == "version":
        sys.exit(version(a.dataset, a.message))


if __name__ == "__main__":
    main()

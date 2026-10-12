"""Static landing page served at '/': what the service is, honest headline numbers, where to try it."""

import html
import json

from .schemas import EXAMPLE_REQUEST

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PayWatch: UPI-style fraud scoring API</title>
<style>
:root{--bg:#0f1419;--fg:#e6edf3;--mut:#9aa7b4;--card:#171e26;--line:#26303b;--acc:#4cc2a6}
@media (prefers-color-scheme:light){:root{--bg:#f6f8fa;--fg:#1f2328;--mut:#59636e;--card:#fff;--line:#d1d9e0;--acc:#0b7a62}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 64px}h1{font-size:2rem;margin:0 0 4px}h2{font-size:1.1rem;margin:32px 0 10px}
.sub{color:var(--mut);margin:0 0 24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.n{font-size:1.5rem;font-weight:650;color:var(--acc)}
.l{color:var(--mut);font-size:.88rem}a{color:var(--acc)}pre{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto;font-size:.85rem}
.note{border-left:3px solid var(--acc);padding:6px 12px;color:var(--mut);margin-top:24px}.btn{display:inline-block;margin:6px 8px 0 0;padding:8px 14px;border:1px solid var(--acc);border-radius:8px;text-decoration:none}
</style></head><body><main>
<h1>PayWatch</h1><p class="sub">Real-time fraud scoring for UPI-style transactions: calibrated probability, action tier and the reasons behind it. Model version <b>__VERSION__</b>.</p>
<a class="btn" href="/docs">Try it: interactive API docs</a><a class="btn" href="/health">/health</a><a class="btn" href="/model-info">/model-info</a>
<a class="btn" href="https://github.com/Sri-Gowtham/paywatch">Source on GitHub</a>
<h2>Held-out test period (time-ordered split, IEEE-CIS data)</h2>
<div class="grid">
<div class="card"><div class="n">0.71</div><div class="l">PR-AUC (0.948 ROC-AUC)</div></div>
<div class="card"><div class="n">64%</div><div class="l">fraud caught at 1% false-positive rate</div></div>
<div class="card"><div class="n">99.5%</div><div class="l">precision on the top 0.5% of alerts (14% recall)</div></div>
<div class="card"><div class="n">~57%</div><div class="l">recall on first-time fraud (no history yet)</div></div>
</div>
<h2>Try a request</h2>
<p class="sub">Open <a href="/docs">/docs</a>, expand <b>POST /predict</b>, press <i>Try it out</i> (the example is pre-filled), then <i>Execute</i>. Or from a terminal:</p>
<pre>curl -X POST __BASE__/predict -H "Content-Type: application/json" -d '__BODY__'</pre>
<div class="note">Honest limits: this is a research demo trained on a public e-commerce fraud dataset, not real UPI traffic. User history lives in
memory and resets when the service restarts or sleeps, so scores start cold. First-time fraud is the hardest case, and fraud labels must
come back (via <code>/feedback</code>) for history features to help. Not for production decisions.</div>
</main></body></html>"""


def landing_html(model_version: str) -> str:
    body = json.dumps(EXAMPLE_REQUEST, separators=(",", ":"))
    return (PAGE.replace("__VERSION__", html.escape(str(model_version)))
            .replace("__BASE__", "https://&lt;this-host&gt;")
            .replace("__BODY__", html.escape(body, quote=False)))

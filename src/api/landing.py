"""HTML landing page served at '/': what the service is, honest headline numbers and a live demo that calls this same API."""

import html

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PayWatch: UPI-style fraud scoring API</title>
<meta name="description" content="Fraud scoring for UPI-style transactions: calibrated probability, action tier and plain-language reasons.">
<style>
:root{--bg:#f9f9f7;--fg:#0b0b0b;--mut:#52514e;--card:#ffffff;--line:#e1e0d9;--acc:#2a78d6;--up:#e34948;--down:#2a78d6;
--g:#0ca30c;--w:#fab219;--s:#ec835a;--c:#d03b3b}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--fg:#f0efec;--mut:#c3c2b7;--card:#1a1a19;--line:#2c2c2a;--acc:#3987e5;--up:#e66767;--down:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:32px 18px 64px}
.hero{border:1px solid var(--line);border-radius:18px;padding:28px 28px 24px;background:linear-gradient(135deg,rgba(42,120,214,.16),rgba(42,120,214,.03) 55%,rgba(235,104,52,.08))}
h1{margin:0;font-size:2.3rem;letter-spacing:-.02em}.lead{margin:8px 0 16px;color:var(--mut);font-size:1.05rem;max-width:720px}
.btn{display:inline-block;padding:9px 16px;border-radius:10px;border:1px solid var(--line);color:var(--fg);text-decoration:none;font-weight:600;margin:0 8px 8px 0;background:var(--card);cursor:pointer;font-size:.95rem}
.btn.p{background:var(--acc);border-color:var(--acc);color:#fff}.btn:disabled{opacity:.6;cursor:wait}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.k{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);font-weight:600}.n{font-size:1.9rem;font-weight:700;letter-spacing:-.02em;line-height:1.2}
.s{font-size:.82rem;color:var(--mut)}h2{font-size:1.15rem;margin:30px 0 10px}
.demo{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:760px){.demo{grid-template-columns:1fr 1.25fr}}
.chip{display:inline-flex;gap:8px;align-items:center;border:1.5px solid;border-radius:999px;padding:6px 16px;font-weight:700;font-size:1.05rem}
.gauge{position:relative;height:60px;margin:22px 4px 6px}.seg{position:absolute;top:20px;height:14px}.tick{position:absolute;top:14px;width:1px;height:26px;background:#888}
.lab{position:absolute;top:40px;font-size:.72rem;color:var(--mut);transform:translateX(-50%)}.mark{position:absolute;top:5px;border-left:7px solid transparent;border-right:7px solid transparent;border-top:12px solid var(--fg);transform:translateX(-50%)}
.val{position:absolute;top:-12px;font-size:.8rem;font-weight:700;transform:translateX(-50%);white-space:nowrap}
.r{margin:0 0 11px}.rt{display:flex;justify-content:space-between;gap:10px;font-size:.9rem}.rb{position:relative;height:10px;margin-top:5px;border-radius:5px;border:1px solid var(--line);background:rgba(127,127,127,.08)}
.rb i{position:absolute;top:0;height:100%;border-radius:4px}.rb u{position:absolute;left:50%;top:-2px;width:1px;height:14px;background:#888}
.ax{display:flex;justify-content:space-between;font-size:.74rem;color:var(--mut)}
pre{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;overflow-x:auto;font-size:.82rem;margin:8px 0}
.note{border-left:4px solid var(--acc);padding:6px 14px;color:var(--mut);margin-top:22px;font-size:.92rem}.muted{color:var(--mut)}
#out.err{color:var(--c);font-weight:600}
</style></head><body><div class="wrap">
<div class="hero"><h1>PayWatch</h1>
<p class="lead">Fraud scoring for UPI-style transactions in about 2 milliseconds: a calibrated probability, an action tier and the reasons behind it in plain language.
Model <b>__VERSION__</b>.</p>
<a class="btn p" href="/docs">Interactive API docs</a><a class="btn" href="https://github.com/Sri-Gowtham/paywatch">Source on GitHub</a>
<a class="btn" href="https://paywatch-r4zappjucbtzxfmxkdjstb5.streamlit.app/">Dashboard</a><a class="btn" href="/health">/health</a><a class="btn" href="/model-info">/model-info</a></div>
<div class="grid">
<div class="card"><div class="k">PR-AUC (test)</div><div class="n">0.724</div><div class="s">a random scorer gets 0.035</div></div>
<div class="card"><div class="k">ROC-AUC (test)</div><div class="n">0.956</div><div class="s">time-ordered, no look-ahead</div></div>
<div class="card"><div class="k">Caught at 1% false alarms</div><div class="n">65%</div><div class="s">of frauds in the held-out period</div></div>
<div class="card"><div class="k">Hard-block precision</div><div class="n">92%</div><div class="s">at 46% recall</div></div></div>
<h2>Try it live</h2>
<div class="demo"><div class="card"><div class="k">Example transaction</div>
<p class="muted" style="margin:6px 0 12px">A complete synthetic transaction, served by this API at <code>/example</code>. Scoring it calls <code>POST /predict</code> with <code>commit: false</code>, so no state changes.</p>
<button class="btn p" id="go">Score the example</button><div class="s" id="status"></div></div>
<div class="card" id="result"><div class="k">Result</div><p class="muted" id="out" style="margin:8px 0 0">Press the button to see the decision, the risk gauge and the reasons.</p></div></div>
<h2>From a terminal</h2>
<pre>curl -s __BASE__/example | curl -s -X POST __BASE__/predict -H "Content-Type: application/json" -d @-</pre>
<div class="note">Honest limits: a research demo trained on a public e-commerce fraud dataset (IEEE-CIS), not real UPI traffic. User history lives in memory and resets when the
service restarts or sleeps, so scores start cold. Recall on first-time fraud is about 57%, and fraud labels must come back (via <code>/feedback</code>) for history features to help.
Not for production decisions.</div></div>
<script>
(function(){
var ST={ALLOW:["#0ca30c","\\u2713","Allow"],SOFT_FLAG:["#fab219","\\u2691","Soft flag"],CHALLENGE:["#ec835a","\\u25C6","Challenge"],HARD_BLOCK:["#d03b3b","\\u25A0","Hard block"]};
function esc(s){var d=document.createElement("div");d.textContent=s;return d.innerHTML}
function pos(v){return Math.sqrt(Math.max(0,Math.min(1,v)))*100}
function gauge(p,th){var t=[th["0.5"],th["0.7"],th["0.9"]],names=["ALLOW","SOFT_FLAG","CHALLENGE","HARD_BLOCK"],cuts=[0,t[0],t[1],t[2],1],h="";
for(var i=0;i<4;i++){var a=pos(cuts[i]),b=pos(cuts[i+1]),c=ST[names[i]][0];h+='<div class="seg" style="left:'+a+'%;width:'+(b-a)+'%;background:'+c+'66;border-left:2px solid '+c+'"></div>';
if(i>0)h+='<div class="tick" style="left:'+a+'%"></div><div class="lab" style="left:'+a+'%">'+Math.round(cuts[i]*100)+'%</div>'}
var m=pos(p);return '<div class="gauge">'+h+'<div class="val" style="left:'+m+'%">'+(p*100).toFixed(1)+'%</div><div class="mark" style="left:'+m+'%"></div></div>'}
function reasons(rs){var mx=1;rs.forEach(function(r){mx=Math.max(mx,Math.abs(r.shap))});mx=0;rs.forEach(function(r){mx=Math.max(mx,Math.abs(r.shap))});mx=mx||1;
var h="";rs.forEach(function(r){var w=Math.abs(r.shap)/mx*50,up=r.shap>0,left=up?50:50-w,col=up?"var(--up)":"var(--down)";
h+='<div class="r"><div class="rt"><span>'+esc(r.text)+'</span><b style="color:'+col+'">'+(up?"+":"")+r.shap.toFixed(2)+'</b></div><div class="rb"><u></u><i style="left:'+left+'%;width:'+w+'%;background:'+col+'"></i></div></div>'});
return h+'<div class="ax"><span>&larr; lowers fraud risk</span><span>raises fraud risk &rarr;</span></div>'}
function j(u,o){return fetch(u,o).then(function(r){if(!r.ok)throw new Error(r.status+" "+r.statusText);return r.json()})}
document.getElementById("go").onclick=function(){var b=this,st=document.getElementById("status"),box=document.getElementById("result");
b.disabled=true;st.textContent="Scoring...";
Promise.all([j("/example"),j("/model-info")]).then(function(v){var body=v[0],th=v[1].tier_thresholds_calibrated;
return j("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(function(o){return [o,th]})}).then(function(v){
var o=v[0],th=v[1],c=ST[o.action];
box.innerHTML='<div class="k">Result</div><div style="margin:8px 0"><span class="chip" style="color:'+c[0]+';border-color:'+c[0]+';background:'+c[0]+'22">'+c[1]+" "+c[2]+'</span> <span class="s">&nbsp;calibrated probability '+(o.calibrated_probability*100).toFixed(1)+'% &middot; '+o.latency_ms.toFixed(0)+' ms</span></div>'+gauge(o.calibrated_probability,th)
+'<div class="s" style="margin-bottom:10px">Square-root scale; coloured bands are the action tiers.</div><div class="k" style="margin-bottom:6px">Why this score</div>'+reasons(o.reasons);
st.textContent="";b.disabled=false;b.textContent="Score again"}).catch(function(e){var out=document.getElementById("out");
if(out){out.className="err";out.textContent="Request failed: "+e.message}st.textContent="";b.disabled=false})}
})();
</script></body></html>"""


def landing_html(model_version: str, base_url: str = "https://<this-host>") -> str:
    return (PAGE.replace("__VERSION__", html.escape(str(model_version)))
            .replace("__BASE__", html.escape(base_url.rstrip("/"), quote=False)))

"""PayWatch dashboard look and feel: CSS, small HTML components and the validated chart palette.

Colours follow the dataviz method: categorical slots validated for colour-vision safety in light and dark (blue, orange, aqua),
a blue<->red diverging pair for risk direction, and fixed status colours that always ship with an icon and a label.
Components emit compact single-line HTML (no indentation or blank lines) so Streamlit's markdown does not turn it into a code block.
"""

import html
from typing import Dict, Iterable, Optional

import streamlit as st

STATUS = {   # action -> (status colour, icon, label)
    "ALLOW": ("#0ca30c", "✓", "Allow"),
    "SOFT_FLAG": ("#fab219", "⚑", "Soft flag"),
    "CHALLENGE": ("#ec835a", "◆", "Challenge"),
    "HARD_BLOCK": ("#d03b3b", "■", "Hard block"),
}
PALETTE = {
    "light": {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "up": "#e34948", "down": "#2a78d6", "muted": "#898781"},
    "dark": {"blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "up": "#e66767", "down": "#3987e5", "muted": "#898781"},
}


def theme_mode() -> str:
    """'light' or 'dark' as the viewer sees it (falls back to light where Streamlit cannot tell)."""
    try:
        t = st.context.theme.type
        return t if t in ("light", "dark") else "light"
    except Exception:
        return "light"


def colors() -> Dict[str, str]:
    return PALETTE[theme_mode()]


CSS = """<style>
:root { --pw-line: rgba(127,127,127,0.28); --pw-card: rgba(127,127,127,0.07); --pw-muted: rgba(127,127,127,0.95); }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 4.5rem !important; max-width: 1180px; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
.pw-hero { border: 1px solid var(--pw-line); border-radius: 16px; padding: 22px 26px; margin-bottom: 14px;
  background: linear-gradient(135deg, rgba(42,120,214,0.14), rgba(42,120,214,0.02) 55%, rgba(235,104,52,0.07)); }
.pw-hero h1 { margin: 0; padding: 0; font-size: 2.1rem; }
.pw-hero p { margin: 6px 0 12px 0; opacity: 0.85; font-size: 1.02rem; }
.pw-badge { display: inline-block; border: 1px solid var(--pw-line); border-radius: 999px; padding: 2px 11px; margin: 0 6px 4px 0;
  font-size: 0.78rem; background: var(--pw-card); }
.pw-card { border: 1px solid var(--pw-line); border-radius: 14px; padding: 14px 16px; background: var(--pw-card); min-height: 128px; box-sizing: border-box; }
.pw-card h4 { margin: 0 0 4px 0; font-size: 0.82rem; font-weight: 600; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.04em; }
.pw-big { font-size: 1.9rem; font-weight: 700; line-height: 1.15; letter-spacing: -0.02em; }
.pw-sub { font-size: 0.82rem; opacity: 0.7; margin-top: 2px; }
.pw-chip { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 6px 16px; font-weight: 700; font-size: 1.05rem;
  border: 1.5px solid; }
.pw-tier { border-left: 5px solid; }
.pw-tier .pw-name { font-weight: 700; font-size: 1.02rem; }
.pw-tier table { width: 100%; font-size: 0.86rem; margin-top: 8px; border-collapse: collapse; border: none; }
.pw-tier td { padding: 3px 0; border: none; border-bottom: 1px solid var(--pw-line); } .pw-tier tr:last-child td { border-bottom: none; }
.pw-tier td:last-child { text-align: right; font-weight: 600; }
.pw-banner { border: 1px solid var(--pw-line); border-left: 5px solid; border-radius: 10px; padding: 10px 14px; margin: 6px 0 12px 0; font-size: 0.95rem; }
.pw-gauge { position: relative; height: 58px; margin: 14px 4px 4px 4px; }
.pw-gauge .seg { position: absolute; top: 18px; height: 14px; }
.pw-gauge .tick { position: absolute; top: 12px; width: 1px; height: 26px; background: rgba(127,127,127,0.9); }
.pw-gauge .lab { position: absolute; top: 38px; font-size: 0.72rem; opacity: 0.75; transform: translateX(-50%); white-space: nowrap; }
.pw-gauge .mark { position: absolute; top: 3px; width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent;
  border-top: 12px solid currentColor; transform: translateX(-50%); }
.pw-gauge .val { position: absolute; top: -14px; font-size: 0.8rem; font-weight: 700; transform: translateX(-50%); white-space: nowrap; }
.pw-steps { display: flex; gap: 10px; flex-wrap: wrap; }
.pw-step { flex: 1 1 200px; border: 1px dashed var(--pw-line); border-radius: 12px; padding: 10px 14px; font-size: 0.9rem; }
.pw-step b { display: block; margin-bottom: 2px; }
.pw-reasons { margin-top: 6px; } .pw-reason { margin: 0 0 12px 0; }
.pw-reason .t { display: flex; justify-content: space-between; gap: 12px; font-size: 0.9rem; } .pw-reason .t b { white-space: nowrap; }
.pw-rbar { position: relative; height: 10px; margin-top: 5px; border-radius: 5px; background: var(--pw-card); border: 1px solid var(--pw-line); }
.pw-rbar i { position: absolute; top: 0; height: 100%; border-radius: 4px; } .pw-rbar u { position: absolute; left: 50%; top: -2px; width: 1px; height: 14px; background: rgba(127,127,127,0.7); }
.pw-axis { display: flex; justify-content: space-between; font-size: 0.74rem; opacity: 0.7; margin-top: -4px; }
[data-testid="stMetric"] { border: 1px solid var(--pw-line); border-radius: 14px; padding: 12px 16px; background: var(--pw-card); }
[data-testid="stMetricLabel"] { opacity: 0.75; }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.02em; }
button[kind="primary"] { font-weight: 600; border-radius: 10px; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--pw-line); }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 16px; font-weight: 600; }
[data-testid="stExpander"] { border-radius: 12px; }
</style>"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badges: Iterable[str]) -> None:
    b = "".join(f'<span class="pw-badge">{html.escape(x)}</span>' for x in badges)
    st.markdown(f'<div class="pw-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{b}</div>', unsafe_allow_html=True)


def card(title: str, big: str, sub: str = "") -> str:
    s = f'<div class="pw-sub">{html.escape(sub)}</div>' if sub else ""
    return f'<div class="pw-card"><h4>{html.escape(title)}</h4><div class="pw-big">{html.escape(big)}</div>{s}</div>'


def chip(action: str) -> str:
    color, icon, label = STATUS.get(action, ("#898781", "•", action))
    return (f'<span class="pw-chip" style="color:{color};border-color:{color};background:{color}22">'
            f'<span>{icon}</span><span style="color:inherit">{html.escape(label)}</span></span>')


def tier_card(action: str, threshold: float, precision: Optional[float], recall: Optional[float], alert_rate: Optional[float]) -> str:
    color, icon, label = STATUS[action]
    fmt = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    return (f'<div class="pw-card pw-tier" style="border-left-color:{color}"><div class="pw-name"><span style="color:{color}">{icon}</span> {html.escape(label)}</div>'
            f'<table><tr><td>Probability at or above</td><td>{threshold:.3f}</td></tr><tr><td>Precision</td><td>{fmt(precision)}</td></tr>'
            f'<tr><td>Fraud caught (recall)</td><td>{fmt(recall)}</td></tr><tr><td>Share of traffic flagged</td><td>{fmt(alert_rate)}</td></tr></table></div>')


def banner(kind: str, text: str) -> None:
    color = {"good": "#0ca30c", "warn": "#fab219", "bad": "#d03b3b", "info": "#2a78d6"}.get(kind, "#2a78d6")
    icon = {"good": "✓", "warn": "⚑", "bad": "■", "info": "ℹ"}.get(kind, "ℹ")
    st.markdown(f'<div class="pw-banner" style="border-left-color:{color}"><b style="color:{color}">{icon}</b>&nbsp; {text}</div>', unsafe_allow_html=True)


def gauge(p: float, thresholds: Dict[str, float]) -> str:
    """Probability on a square-root scale (so 2% and 70% both read), tier bands in status colours, a marker at p."""
    pos = lambda v: max(0.0, min(1.0, v)) ** 0.5 * 100  # noqa: E731
    t1, t2, t3 = (thresholds.get(k) for k in ("0.5", "0.7", "0.9"))
    cuts = [0.0] + [t for t in (t1, t2, t3) if t is not None] + [1.0]
    names = ["ALLOW"] + [n for n, t in (("SOFT_FLAG", t1), ("CHALLENGE", t2), ("HARD_BLOCK", t3)) if t is not None]
    segs, ticks = [], []
    for i, name in enumerate(names):
        a, b = pos(cuts[i]), pos(cuts[i + 1])
        c = STATUS[name][0]
        segs.append(f'<div class="seg" style="left:{a:.2f}%;width:{b - a:.2f}%;background:{c}66;border-left:2px solid {c}"></div>')
        if i > 0:
            ticks.append(f'<div class="tick" style="left:{a:.2f}%"></div><div class="lab" style="left:{a:.2f}%">{cuts[i]:.0%}</div>')
    m = pos(p)
    return (f'<div class="pw-gauge">{"".join(segs)}{"".join(ticks)}<div class="val" style="left:{m:.2f}%">{p:.1%}</div>'
            f'<div class="mark" style="left:{m:.2f}%"></div></div>')


def reasons_list(reasons) -> str:
    """Top reasons as rows: plain-language text, a bar centred on zero (red = raises risk, blue = lowers it) and the signed size."""
    c = colors()
    mx = max((abs(r["shap"]) for r in reasons), default=1.0) or 1.0
    rows = []
    for r in reasons:
        w = abs(r["shap"]) / mx * 50
        up = r["shap"] > 0
        left, color = (50.0, c["up"]) if up else (50.0 - w, c["down"])
        rows.append(f'<div class="pw-reason"><div class="t"><span>{html.escape(r["text"])}</span><b style="color:{color}">{r["shap"]:+.2f}</b></div>'
                    f'<div class="pw-rbar"><u></u><i style="left:{left:.1f}%;width:{w:.1f}%;background:{color}"></i></div></div>')
    return (f'<div class="pw-reasons">{"".join(rows)}<div class="pw-axis"><span>&larr; lowers fraud risk</span><span>raises fraud risk &rarr;</span></div></div>')

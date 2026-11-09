"""Look & feel for the CreditAlo dashboard: CSS, Plotly template, small HTML components."""
import plotly.graph_objects as go
import plotly.io as pio

TEAL, VIOLET, PINK, AMBER, SKY, GREEN, RED = "#00E5C7", "#7C5CFF", "#FF3D81", "#FFB020", "#38BDF8", "#22C55E", "#EF4444"
MUTED, TEXT = "#8B97B5", "#E6EAF5"
MODEL_COLORS = {"Logistic Regression": SKY, "Random Forest": AMBER, "LightGBM": TEAL}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -8%, rgba(124,92,255,.26), transparent 60%),
    radial-gradient(900px 520px at 100% 0%, rgba(0,229,199,.16), transparent 55%),
    radial-gradient(700px 500px at 50% 110%, rgba(255,61,129,.10), transparent 60%), #070B17;
}
header[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1420px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0B1226,#080D1D); border-right: 1px solid rgba(255,255,255,.06); }

/* ---------- hero ---------- */
.tl-hero { position: relative; overflow: hidden; border-radius: 22px; padding: 26px 30px; margin-bottom: 14px;
  background: linear-gradient(120deg, rgba(124,92,255,.30), rgba(0,229,199,.16) 55%, rgba(255,61,129,.18));
  border: 1px solid rgba(255,255,255,.12); box-shadow: 0 20px 60px rgba(0,0,0,.45); }
.tl-hero::after { content:""; position:absolute; inset:-40% -10% auto auto; width:520px; height:520px; border-radius:50%;
  background: radial-gradient(circle, rgba(0,229,199,.28), transparent 65%); animation: float 9s ease-in-out infinite; }
@keyframes float { 0%,100% { transform: translate(0,0);} 50% { transform: translate(-40px,30px);} }
.tl-row { display:flex; align-items:center; gap:18px; position:relative; z-index:2; flex-wrap:wrap; }
.tl-logo { width:62px; height:62px; border-radius:18px; display:flex; align-items:center; justify-content:center; font-size:32px;
  background: linear-gradient(135deg,#00E5C7,#7C5CFF); box-shadow: 0 8px 30px rgba(0,229,199,.35); }
.tl-title { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.35rem; line-height:1.05; letter-spacing:-.02em;
  background: linear-gradient(90deg,#fff,#9BF3E6 60%,#C7B8FF); -webkit-background-clip:text; background-clip:text; color:transparent; }
.tl-tag { color:#C9D3F0; font-size:.98rem; margin-top:4px; }
.tl-chips { display:flex; gap:8px; flex-wrap:wrap; margin-left:auto; }
.tl-chip { padding:6px 12px; border-radius:999px; font-size:.78rem; font-weight:600; color:#E6EAF5;
  background: rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); backdrop-filter: blur(8px); }
.tl-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; }
.tl-on { background:#22C55E; box-shadow:0 0 0 0 rgba(34,197,94,.7); animation:pulse 1.8s infinite; }
.tl-off { background:#EF4444; }
@keyframes pulse { 70% { box-shadow:0 0 0 9px rgba(34,197,94,0);} 100% { box-shadow:0 0 0 0 rgba(34,197,94,0);} }

/* ---------- sticky feature bar (tabs) ---------- */
div[data-baseweb="tab-list"] { position: sticky; top: 0; z-index: 999; gap: 6px; padding: 7px; margin-bottom: 8px;
  background: rgba(8,13,29,.82); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px; box-shadow: 0 10px 30px rgba(0,0,0,.45); }
button[data-baseweb="tab"] { flex: 1; height: 46px; border-radius: 13px; color: #9AA7C7; font-weight: 600; font-size: .95rem;
  background: transparent; transition: all .2s ease; }
button[data-baseweb="tab"]:hover { color:#fff; background: rgba(255,255,255,.06); }
button[data-baseweb="tab"][aria-selected="true"] { color:#04110F; background: linear-gradient(90deg,#00E5C7,#7CF0DD); box-shadow: 0 6px 22px rgba(0,229,199,.35); }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }

/* ---------- cards ---------- */
.tl-kpi { position:relative; border-radius:18px; padding:16px 18px; height:100%;
  background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); border:1px solid rgba(255,255,255,.09); overflow:hidden; }
.tl-kpi::before { content:""; position:absolute; left:0; top:0; right:0; height:3px; background: var(--accent); }
.tl-kpi .l { color:#97A3C2; font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; font-weight:600; }
.tl-kpi .v { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#fff; margin-top:6px; line-height:1.1; }
.tl-kpi .s { color:#7F8CAD; font-size:.8rem; margin-top:4px; }
.tl-card { border-radius:18px; padding:16px 18px; background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); }
.tl-h { font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700; margin:8px 0 2px; color:#fff; }
.tl-sub { color:#8B97B5; font-size:.88rem; margin-bottom:10px; }
.tl-pill { display:inline-block; padding:5px 14px; border-radius:999px; font-weight:700; font-size:.85rem; }
.tl-rc { border-radius:14px; padding:11px 14px; margin-bottom:8px; border:1px solid rgba(255,255,255,.08); }
.tl-rc.r { background: linear-gradient(90deg, rgba(255,61,129,.18), rgba(255,61,129,.04)); border-left:4px solid #FF3D81; }
.tl-rc.p { background: linear-gradient(90deg, rgba(0,229,199,.16), rgba(0,229,199,.03)); border-left:4px solid #00E5C7; }
.tl-rc b { color:#fff; } .tl-rc span { color:#9AA7C7; font-size:.78rem; }
.tl-check { display:flex; gap:10px; align-items:flex-start; padding:10px 12px; border-radius:12px; margin-bottom:7px;
  background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07); font-size:.9rem; }
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; border: 1px solid rgba(0,229,199,.45); }
</style>
"""


def html(s: str) -> str:
    """Collapse indentation so Streamlit's markdown never turns the HTML into a code block."""
    return "".join(line.strip() for line in s.strip().splitlines())


def kpi(label: str, value: str, sub: str = "", accent: str = TEAL) -> str:
    return html(f'<div class="tl-kpi" style="--accent:{accent}"><div class="l">{label}</div>'
                f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def section(title: str, sub: str = "") -> str:
    return html(f'<div class="tl-h">{title}</div><div class="tl-sub">{sub}</div>')


def pill(text: str, colour: str) -> str:
    return html(f'<span class="tl-pill" style="background:{colour}26;color:{colour};border:1px solid {colour}66">{text}</span>')


def check(ok: bool, title: str, detail: str) -> str:
    icon = "✅" if ok else "⚠️"
    return html(f'<div class="tl-check"><div>{icon}</div><div><b>{title}</b><br>'
                f'<span style="color:#8B97B5">{detail}</span></div></div>')


def register_plotly_template() -> None:
    pio.templates["creditalo"] = go.layout.Template(layout=go.Layout(
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[TEAL, VIOLET, PINK, AMBER, SKY, GREEN],
        xaxis=dict(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)", linecolor="rgba(255,255,255,.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)", linecolor="rgba(255,255,255,.12)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor="#0F1730", font=dict(color="#fff")),
    ))
    pio.templates.default = "creditalo"

"""
VetTriageEnv — Gradio UI (vibrant edition)
Mounted at /ui on the FastAPI server.
"""
from __future__ import annotations

import json
from typing import Optional

import gradio as gr

from vettriagevenv.env import VetTriageEnv
from vettriagevenv.models import Action
from vettriagevenv.tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Shared env instance
# ---------------------------------------------------------------------------
_env = VetTriageEnv(max_total_steps=100)
_state: dict = {}

TASK_CHOICES = list(TASK_REGISTRY.keys())

TOOL_CHOICES = [
    "check_vitals", "physical_exam", "run_bloodwork", "run_imaging",
    "collect_result", "place_iv_access", "administer_fluid_bolus",
    "give_medication", "oxygen_therapy", "perform_procedure",
    "contact_owner", "consult_specialist", "decide_triage_route", "make_disposition",
]

TOOL_PARAM_HINTS = {
    "check_vitals":           '{"systems": ["cardiovascular", "respiratory", "temperature", "pain", "mentation"]}',
    "physical_exam":          '{"region": "abdomen"}',
    "run_bloodwork":          '{"panel": "cbc"}',
    "run_imaging":            '{"modality": "radiograph", "region": "thorax"}',
    "collect_result":         '{"job_id": "<from pending results>"}',
    "place_iv_access":        '{"site": "cephalic"}',
    "administer_fluid_bolus": '{"fluid_type": "crystalloid", "dose_ml_kg": 10}',
    "give_medication":        '{"drug": "morphine", "dose": "0.2mg/kg", "route": "iv"}',
    "oxygen_therapy":         '{"method": "mask"}',
    "perform_procedure":      '{"procedure": "gastric_decompression"}',
    "contact_owner":          '{"message": "Your pet needs urgent treatment."}',
    "consult_specialist":     '{"specialty": "surgery", "question": "Surgical consult needed?"}',
    "decide_triage_route":    '{"route": "immediate_resuscitation"}',
    "make_disposition":       '{"disposition": "admit_icu"}',
}

# Emojis and colours per task
TASK_META = {
    "easy_gdv": {
        "emoji": "🐕",
        "label": "GDV Emergency",
        "badge": "🟢 EASY",
        "img": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400&q=80",
        "desc": "German Shepherd with acute gastric dilatation-volvulus. Clear signals — baseline test.",
        "tip": "💡 Check cardiovascular vitals → abdominal exam → immediate_resuscitation → admit_icu",
    },
    "medium_hcm_cat": {
        "emoji": "🐈",
        "label": "HCM Cat",
        "badge": "🟡 MEDIUM",
        "img": "https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=400&q=80",
        "desc": "Dyspnoeic cat with hypertrophic cardiomyopathy. Time-pressure dilemma.",
        "tip": "💡 Do NOT give crystalloid fluids. Use furosemide. Thoracocentesis if pleural effusion.",
    },
    "hard_imha_budget": {
        "emoji": "🩸",
        "label": "IMHA Budget",
        "badge": "🔴 HARD",
        "img": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&q=80",
        "desc": "Immune haemolytic anaemia with a ₹38,000 hard budget. Every rupee counts.",
        "tip": "💡 Contact owner first to learn the budget. CBC + lactate only — skip expensive imaging.",
    },
    "hard_polytrauma": {
        "emoji": "🚑",
        "label": "Polytrauma HBC",
        "badge": "🔴 HARD",
        "img": "https://images.unsplash.com/photo-1601758174114-e711687b4283?w=400&q=80",
        "desc": "Hit-by-car dog. Mid-episode seizure. Multiple injuries.",
        "tip": "💡 Image thorax + abdomen urgently. Prefer colloid over crystalloid. Watch for seizure.",
    },
    "hard_stochastic_pancreatitis": {
        "emoji": "⚡",
        "label": "Stochastic Pancreatitis",
        "badge": "🔴 HARD",
        "img": "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=400&q=80",
        "desc": "Uncooperative Border Collie. Actions fail silently — check action_succeeded every step.",
        "tip": "💡 If a tool fails silently, switch route/method/site on the next attempt.",
    },
    "hard_parvovirus_day1": {
        "emoji": "🦠",
        "label": "Parvo Day 1",
        "badge": "🔴 HARD",
        "img": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400&q=80",
        "desc": "Unvaccinated puppy. SNAP test has 25% false-negative rate on day 1.",
        "tip": "💡 Negative SNAP ≠ no parvo. Treat on clinical signs: bloody diarrhoea + leukopenia.",
    },
    "hard_nosocomial_chf_ward": {
        "emoji": "🏥",
        "label": "Nosocomial CHF",
        "badge": "🔴 HARD",
        "img": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&q=80",
        "desc": "CHF dog already 20h into ward stay. Infection risk escalates every 24h.",
        "tip": "💡 Discharge ASAP — infection probability: 10% at 24h, 25% at 48h, 50% at 72h.",
    },
}

CSS = """
/* ── Base ── */
body, .gradio-container { background: #0a0f1e !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }

/* ── Header banner ── */
#header-banner {
    background: linear-gradient(135deg, #1a1f3a 0%, #0d1b2a 50%, #1a0a2e 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 8px;
    text-align: center;
}
#header-banner h1 { font-size: 2.4rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#header-banner p  { color: #94a3b8; margin: 6px 0 0; font-size: 1rem; }

/* ── Cards ── */
.card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 8px;
}

/* ── Task image ── */
#task-img img { border-radius: 12px; object-fit: cover; width: 100%; max-height: 180px; }

/* ── Badges ── */
.badge-easy   { background: #064e3b; color: #6ee7b7; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700; }
.badge-medium { background: #78350f; color: #fcd34d; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700; }
.badge-hard   { background: #7f1d1d; color: #fca5a5; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700; }

/* ── Observation panel ── */
#obs-panel {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    min-height: 280px;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* ── Status panel ── */
#status-panel {
    background: #0f172a;
    border: 1px solid #2d1b69;
    border-radius: 12px;
    padding: 16px;
    min-height: 160px;
}

/* ── History panel ── */
#history-panel {
    background: #0f172a;
    border: 1px solid #14532d;
    border-radius: 12px;
    padding: 16px;
    min-height: 120px;
    font-size: 0.82rem;
}

/* ── Buttons ── */
#start-btn { background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-size: 1rem !important; font-weight: 700 !important; padding: 12px !important; }
#step-btn  { background: linear-gradient(90deg, #059669, #0891b2) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-size: 1rem !important; font-weight: 700 !important; padding: 12px !important; }
#start-btn:hover { background: linear-gradient(90deg, #1d4ed8, #6d28d9) !important; }
#step-btn:hover  { background: linear-gradient(90deg, #047857, #0e7490) !important; }

/* ── Inputs ── */
.gr-textbox textarea, .gr-dropdown select {
    background: #1e293b !important; color: #e2e8f0 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
}
label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* ── Severity bar ── */
#severity-bar { height: 8px; border-radius: 4px; margin: 6px 0;
    background: linear-gradient(90deg, #22c55e, #eab308, #ef4444); }

/* ── Score panel ── */
#score-display {
    background: linear-gradient(135deg, #064e3b, #0c4a6e);
    border: 1px solid #065f46;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

/* ── Section labels ── */
.section-label { color: #60a5fa; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
"""


def _severity_bar(severity: Optional[float]) -> str:
    if severity is None:
        return ""
    pct = int(severity * 100)
    colour = "#22c55e" if pct < 40 else "#eab308" if pct < 65 else "#ef4444"
    return f"""
<div style="margin:8px 0">
  <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94a3b8">
    <span>Patient Severity</span><span style="color:{colour};font-weight:700">{pct}%</span>
  </div>
  <div style="height:8px;background:#1e293b;border-radius:4px;overflow:hidden">
    <div style="width:{pct}%;height:100%;background:{colour};border-radius:4px;transition:width 0.4s"></div>
  </div>
</div>"""


def _fmt_obs(obs, severity: Optional[float] = None) -> str:
    lines = []

    # Phase / step strip
    phase_colour = {"triage": "#60a5fa", "stabilisation": "#f59e0b",
                    "monitoring": "#34d399", "disposition": "#a78bfa"}.get(obs.phase, "#94a3b8")
    lines.append(
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px">'
        f'<span style="background:#1e293b;border:1px solid {phase_colour};color:{phase_colour};'
        f'border-radius:6px;padding:2px 10px;font-size:0.8rem;font-weight:700">📍 {obs.phase.upper()}</span>'
        f'<span style="color:#94a3b8;font-size:0.85rem">Step {obs.step}/{obs.phase_step_limit}</span>'
        f'<span style="color:#94a3b8;font-size:0.85rem">⏱ {obs.sim_time_hours:.1f}h elapsed</span>'
    )
    if obs.budget_limit is not None:
        remaining = obs.budget_remaining if obs.budget_remaining is not None else obs.budget_limit - obs.budget_spent
        col = "#ef4444" if remaining < obs.budget_limit * 0.2 else "#34d399"
        lines.append(f'<span style="color:{col};font-size:0.85rem">💰 ₹{remaining:.0f} left</span>')
    lines.append("</div>")

    # Severity bar
    if severity is not None:
        lines.append(_severity_bar(severity))

    # Failure alert
    if not obs.action_succeeded:
        lines.append(
            f'<div style="background:#450a0a;border:1px solid #991b1b;border-radius:8px;'
            f'padding:8px 12px;margin:8px 0;color:#fca5a5;font-size:0.85rem">'
            f'⚠️ <strong>ACTION FAILED:</strong> {obs.latest_clinical_event}</div>'
        )
    elif obs.latest_clinical_event:
        lines.append(
            f'<div style="background:#0c2d48;border:1px solid #1e40af;border-radius:8px;'
            f'padding:8px 12px;margin:8px 0;color:#93c5fd;font-size:0.85rem">'
            f'ℹ️ {obs.latest_clinical_event}</div>'
        )

    # Events
    for e in obs.events:
        lines.append(
            f'<div style="background:#2d1b00;border:1px solid #92400e;border-radius:8px;'
            f'padding:8px 12px;margin:6px 0;color:#fcd34d;font-size:0.85rem">🚨 {e}</div>'
        )

    def section(title, icon, content):
        return (
            f'<div style="margin:10px 0">'
            f'<div style="color:#60a5fa;font-size:0.72rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:4px">{icon} {title}</div>'
            f'<div style="background:#0d1b2a;border-radius:8px;padding:10px;font-size:0.85rem;color:#cbd5e1">'
            f'{content}</div></div>'
        )

    # Vitals
    if obs.vitals:
        v = obs.vitals if isinstance(obs.vitals, dict) else obs.vitals.model_dump()
        items = []
        icons = {"heart_rate": "❤️", "respiratory_rate": "🫁", "temperature": "🌡️",
                 "spo2": "💉", "systolic_bp": "🩺", "mentation": "🧠",
                 "mucous_membrane_color": "👄", "capillary_refill_time": "⏱️", "pain_score": "😣"}
        for k, val in v.items():
            if val is not None and k not in ("systems_checked",):
                ico = icons.get(k, "•")
                label = k.replace("_", " ").title()
                items.append(f'<span style="margin-right:16px">{ico} <strong>{label}:</strong> {val}</span>')
        lines.append(section("Vitals", "🔬", " ".join(items) or "—"))

    # Exam
    if obs.physical_exam_findings:
        rows = "".join(
            f'<div><strong>{r}:</strong> {f}</div>'
            for r, f in obs.physical_exam_findings.items()
        )
        lines.append(section("Physical Exam", "🩻", rows))

    # Labs
    if obs.lab_results:
        rows = []
        for panel, result in obs.lab_results.items():
            interp = result.get("interpretation", str(result)) if isinstance(result, dict) else str(result)
            rows.append(f'<div><strong>{panel}:</strong> {interp}</div>')
        lines.append(section("Lab Results", "🧪", "".join(rows)))

    # Imaging
    if obs.imaging_results:
        rows = []
        for key, result in obs.imaging_results.items():
            interp = result.get("interpretation", str(result)) if isinstance(result, dict) else str(result)
            rows.append(f'<div><strong>{key}:</strong> {interp}</div>')
        lines.append(section("Imaging", "📷", "".join(rows)))

    # Pending
    if obs.pending_results:
        rows = []
        for p in obs.pending_results:
            jid = p.get("job_id") if isinstance(p, dict) else p.job_id
            panel = p.get("panel_or_modality") if isinstance(p, dict) else p.panel_or_modality
            eta = p.get("eta_steps") if isinstance(p, dict) else p.eta_steps
            rows.append(f'<div>📋 <code>{jid}</code> — {panel} (~{eta} steps)</div>')
        lines.append(section("Pending Results", "⏳", "".join(rows)))

    # Monitoring trends
    if obs.monitoring_trends:
        rows = []
        for k, v in obs.monitoring_trends.items():
            colour = "#ef4444" if "HIGH" in str(v) or "ACTIVE" in str(v) else \
                     "#f59e0b" if "moderate" in str(v).lower() else "#34d399"
            rows.append(f'<div style="color:{colour}"><strong>{k}:</strong> {v}</div>')
        lines.append(section("Monitoring Trends", "📈", "".join(rows)))

    # Specialist opinion
    if obs.specialist_opinion:
        lines.append(section("Specialist Opinion", "👨‍⚕️", obs.specialist_opinion))

    # Available tools
    if obs.available_tools:
        tools_html = " ".join(
            f'<code style="background:#1e293b;border:1px solid #334155;border-radius:4px;'
            f'padding:1px 6px;font-size:0.78rem;color:#a5f3fc">{t}</code>'
            for t in obs.available_tools
        )
        lines.append(f'<div style="margin-top:10px">{tools_html}</div>')

    return "\n".join(lines)


def _fmt_history(obs) -> str:
    if not obs.action_history:
        return '<span style="color:#475569;font-style:italic">No actions yet.</span>'
    rows = []
    for h in obs.action_history[-8:]:
        colour = "#fca5a5" if "[FAILED]" in h else "#86efac"
        rows.append(f'<div style="color:{colour};margin:2px 0;font-family:monospace;font-size:0.8rem">{h}</div>')
    return "\n".join(rows)


def _task_card(task_id: str) -> str:
    m = TASK_META.get(task_id, {})
    badge = m.get("badge", "")
    desc  = m.get("desc", "")
    tip   = m.get("tip", "")
    emoji = m.get("emoji", "🐾")
    return (
        f'<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px">'
        f'<div style="font-size:2rem;margin-bottom:6px">{emoji}</div>'
        f'<div style="margin-bottom:6px"><span style="background:#1e293b;border-radius:6px;'
        f'padding:2px 10px;font-size:0.8rem;font-weight:700;color:#e2e8f0">{badge}</span></div>'
        f'<div style="color:#cbd5e1;font-size:0.88rem;margin:6px 0">{desc}</div>'
        f'<div style="color:#fbbf24;font-size:0.82rem;margin-top:8px">{tip}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def on_task_change(task_id: str):
    m = TASK_META.get(task_id, {})
    img = m.get("img", "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400&q=80")
    card = _task_card(task_id)
    return img, card


def do_reset(task_id: str):
    global _state
    obs = _env.reset(task_id=task_id, seed=42)
    _state = {"obs": obs, "done": False, "score": None, "rewards": [], "step": 0}

    patient_html = (
        f'<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px">'
        f'<div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-bottom:8px">'
        f'{TASK_META.get(task_id,{}).get("emoji","🐾")} {obs.species.title()} — {obs.breed}</div>'
        f'<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:4px">'
        f'🎂 {obs.age_years:.1f}yr &nbsp;|&nbsp; ⚖️ {obs.weight_kg:.1f}kg &nbsp;|&nbsp; {obs.sex}</div>'
        f'<div style="background:#0f172a;border-radius:8px;padding:10px;color:#cbd5e1;'
        f'font-size:0.85rem;margin-top:8px;line-height:1.6">'
        f'<strong style="color:#60a5fa">Presenting Complaint:</strong><br>{obs.presenting_complaint}</div>'
        f'</div>'
    )

    obs_html = _fmt_obs(obs)
    hist_html = _fmt_history(obs)

    status_html = (
        f'<div style="background:#064e3b;border:1px solid #065f46;border-radius:10px;'
        f'padding:12px;color:#6ee7b7;font-weight:600">▶ Episode started! Choose a tool and take your first action.</div>'
    )

    return (
        patient_html,
        obs_html,
        hist_html,
        status_html,
        gr.update(interactive=True),
    )


def on_tool_change(tool: str):
    return TOOL_PARAM_HINTS.get(tool, "{}")


def do_step(tool: str, params_str: str, reasoning: str):
    global _state
    if not _state or _state.get("done"):
        return (
            _fmt_obs(_state["obs"]) if _state.get("obs") else "",
            _fmt_history(_state["obs"]) if _state.get("obs") else "",
            '<div style="color:#f87171">⚠️ No active episode. Click Start Episode first.</div>',
            gr.update(interactive=False),
        )

    try:
        params = json.loads(params_str) if params_str.strip() else {}
    except json.JSONDecodeError as e:
        return (
            _fmt_obs(_state["obs"]),
            _fmt_history(_state["obs"]),
            f'<div style="color:#f87171">❌ Invalid JSON: {e}</div>',
            gr.update(interactive=True),
        )

    action = Action(tool=tool, parameters=params, reasoning=reasoning or None)
    obs, reward, done, info = _env.step(action)

    # Try to get severity from env state
    severity = None
    try:
        s = _env.state()
        severity = s.get("internal_state", {}).get("patient", {}).get("severity")
    except Exception:
        pass

    _state["obs"] = obs
    _state["rewards"].append(reward.value)
    _state["step"] += 1

    reward_colour = "#34d399" if reward.value >= 0 else "#f87171"
    status_html = (
        f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:12px">'
        f'<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:8px">'
        f'<span style="color:#94a3b8;font-size:0.85rem">Step <strong style="color:#e2e8f0">{_state["step"]}</strong></span>'
        f'<span style="color:#94a3b8;font-size:0.85rem">Tool: <code style="color:#a5f3fc">{tool}</code></span>'
        f'<span style="color:{reward_colour};font-weight:700;font-size:0.95rem">Reward: {reward.value:+.3f}</span>'
        f'</div>'
        f'<div style="color:#94a3b8;font-size:0.82rem;font-style:italic">{reward.message}</div>'
        f'</div>'
    )

    if done:
        _state["done"] = True
        grade = info.get("grade", 0.0)
        passed = info.get("passed", False)
        feedback = info.get("grade_feedback", [])
        _state["score"] = grade

        grade_colour = "#34d399" if grade >= 0.7 else "#f59e0b" if grade >= 0.5 else "#f87171"
        pass_badge = (
            '<span style="background:#064e3b;color:#6ee7b7;border-radius:6px;padding:2px 10px;font-weight:700">✅ PASSED</span>'
            if passed else
            '<span style="background:#7f1d1d;color:#fca5a5;border-radius:6px;padding:2px 10px;font-weight:700">❌ FAILED</span>'
        )
        fb_rows = "".join(f'<li style="color:#cbd5e1;font-size:0.83rem">{f}</li>' for f in feedback)

        status_html = (
            f'<div style="background:linear-gradient(135deg,#064e3b,#0c4a6e);'
            f'border:1px solid #065f46;border-radius:12px;padding:16px;text-align:center">'
            f'<div style="font-size:1.5rem;font-weight:800;color:{grade_colour};margin-bottom:6px">'
            f'🏁 Episode Complete!</div>'
            f'<div style="font-size:2rem;font-weight:900;color:{grade_colour}">{grade:.3f}</div>'
            f'<div style="color:#94a3b8;font-size:0.85rem;margin:4px 0">Final Score</div>'
            f'<div style="margin:8px 0">{pass_badge}</div>'
            f'<ul style="text-align:left;margin-top:12px;padding-left:16px">{ fb_rows }</ul>'
            f'</div>'
        )
        return (
            _fmt_obs(obs, severity),
            _fmt_history(obs),
            status_html,
            gr.update(interactive=False),
        )

    return (
        _fmt_obs(obs, severity),
        _fmt_history(obs),
        status_html,
        gr.update(interactive=True),
    )


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    default_task = "easy_gdv"
    default_meta = TASK_META[default_task]

    with gr.Blocks(title="🐾 VetTriageEnv") as demo:
        gr.HTML(f"<style>{CSS}</style>")

        # ── Header ──
        gr.HTML("""
        <div id="header-banner">
          <h1>🐾 VetTriageEnv</h1>
          <p>AI Veterinary Triage — Interactive Benchmark Environment</p>
          <p style="font-size:0.78rem;color:#475569;margin-top:4px">
            Diagnose · Stabilise · Disposition · Score
          </p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left column — task selector + patient ──
            with gr.Column(scale=1, min_width=280):

                gr.HTML('<div class="section-label">📋 Select Task</div>')
                task_dd = gr.Dropdown(
                    choices=TASK_CHOICES, value=default_task,
                    label="", show_label=False, container=False,
                )

                task_img = gr.Image(
                    value=default_meta["img"],
                    show_label=False,
                    height=160, elem_id="task-img", container=False,
                )

                task_card_html = gr.HTML(_task_card(default_task))

                start_btn = gr.Button("▶  Start Episode", elem_id="start-btn", size="lg")

                gr.HTML('<div class="section-label" style="margin-top:12px">🐕 Patient</div>')
                patient_html = gr.HTML(
                    '<div style="color:#475569;font-style:italic;font-size:0.85rem">'
                    'Start an episode to see patient details.</div>'
                )

            # ── Right column — observation + history ──
            with gr.Column(scale=2):

                gr.HTML('<div class="section-label">🔭 Observation</div>')
                obs_html = gr.HTML(
                    '<div id="obs-panel" style="color:#475569;font-style:italic">'
                    'Start an episode to see the clinical picture.</div>',
                    elem_id="obs-panel",
                )

                gr.HTML('<div class="section-label" style="margin-top:12px">📜 Action History</div>')
                history_html = gr.HTML(
                    '<div id="history-panel" style="color:#475569;font-style:italic">'
                    'No actions yet.</div>',
                    elem_id="history-panel",
                )

        # ── Action row ──
        gr.HTML('<hr style="border-color:#1f2937;margin:16px 0">')
        gr.HTML('<div class="section-label">⚡ Take an Action</div>')

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                tool_dd = gr.Dropdown(
                    choices=TOOL_CHOICES, value="check_vitals", label="Tool"
                )
                params_box = gr.Textbox(
                    value=TOOL_PARAM_HINTS["check_vitals"],
                    label="Parameters (JSON)", lines=3,
                )
                reasoning_box = gr.Textbox(
                    placeholder="Optional: explain your clinical reasoning…",
                    label="Reasoning", lines=2,
                )
                step_btn = gr.Button(
                    "⚡  Take Action", elem_id="step-btn",
                    size="lg", interactive=False,
                )

            with gr.Column(scale=1):
                gr.HTML('<div class="section-label">📊 Step Result</div>')
                status_html = gr.HTML(
                    '<div id="status-panel" style="color:#475569;font-style:italic">'
                    'Results will appear here.</div>',
                    elem_id="status-panel",
                )

        # ── Footer ──
        gr.HTML("""
        <div style="text-align:center;color:#334155;font-size:0.75rem;margin-top:20px;padding:12px;
             border-top:1px solid #1f2937">
          VetTriageEnv · OpenEnv Hackathon · 7 tasks · Easy → Hard
        </div>
        """)

        # ── Wiring ──
        task_dd.change(
            fn=on_task_change, inputs=task_dd,
            outputs=[task_img, task_card_html],
        )
        start_btn.click(
            fn=do_reset, inputs=task_dd,
            outputs=[patient_html, obs_html, history_html, status_html, step_btn],
        )
        tool_dd.change(
            fn=on_tool_change, inputs=tool_dd, outputs=params_box,
        )
        step_btn.click(
            fn=do_step, inputs=[tool_dd, params_box, reasoning_box],
            outputs=[obs_html, history_html, status_html, step_btn],
        )

    return demo

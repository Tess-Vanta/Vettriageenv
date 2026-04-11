"""
VetTriageEnv — Gradio UI (vibrant edition)
Mounted at /ui on the FastAPI server.
"""
from __future__ import annotations

import json
import os
import textwrap
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

# SVG illustrations per task — inline, no external requests, always render
TASK_SVGS = {
    "easy_gdv": """<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="180" fill="#0d1b2a"/>
  <ellipse cx="200" cy="110" rx="90" ry="55" fill="#1e3a5f" stroke="#3b82f6" stroke-width="2"/>
  <ellipse cx="200" cy="105" rx="75" ry="42" fill="#1e40af" opacity="0.5"/>
  <circle cx="130" cy="75" r="28" fill="#1e3a5f" stroke="#3b82f6" stroke-width="2"/>
  <ellipse cx="130" cy="62" rx="12" ry="8" fill="#2563eb"/>
  <ellipse cx="122" cy="58" rx="5" ry="7" fill="#1e3a5f" stroke="#3b82f6" stroke-width="1.5"/>
  <ellipse cx="138" cy="58" rx="5" ry="7" fill="#1e3a5f" stroke="#3b82f6" stroke-width="1.5"/>
  <circle cx="122" cy="76" r="3" fill="#93c5fd"/>
  <circle cx="138" cy="76" r="3" fill="#93c5fd"/>
  <path d="M115 83 Q130 88 145 83" stroke="#93c5fd" stroke-width="2" fill="none"/>
  <rect x="270" y="85" width="8" height="50" rx="4" fill="#1e3a5f" stroke="#3b82f6" stroke-width="1.5"/>
  <rect x="285" y="70" width="8" height="65" rx="4" fill="#1e3a5f" stroke="#3b82f6" stroke-width="1.5"/>
  <rect x="300" y="90" width="8" height="45" rx="4" fill="#1e3a5f" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="200" y="168" text-anchor="middle" fill="#60a5fa" font-size="13" font-family="monospace" font-weight="bold">GDV — ACUTE ABDOMEN</text>
  <text x="310" y="60" fill="#ef4444" font-size="11" font-family="monospace">BLOAT ↑</text>
  <path d="M185 90 Q200 70 215 90" stroke="#ef4444" stroke-width="2" fill="none" stroke-dasharray="4"/>
</svg>""",

    "medium_hcm_cat": """<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="180" fill="#0d1b2a"/>
  <ellipse cx="190" cy="115" rx="70" ry="45" fill="#1a1f3a" stroke="#a78bfa" stroke-width="2"/>
  <circle cx="145" cy="78" r="30" fill="#1a1f3a" stroke="#a78bfa" stroke-width="2"/>
  <polygon points="133,52 140,35 147,52" fill="#a78bfa"/>
  <polygon points="148,52 155,35 162,52" fill="#a78bfa"/>
  <circle cx="137" cy="80" r="4" fill="#7c3aed"/>
  <circle cx="153" cy="80" r="4" fill="#7c3aed"/>
  <path d="M130 90 Q145 96 160 90" stroke="#c4b5fd" stroke-width="2" fill="none"/>
  <path d="M115 78 L95 72 M115 82 L93 82 M115 86 L95 90" stroke="#a78bfa" stroke-width="1.5"/>
  <path d="M165 78 L185 72 M165 82 L187 82 M165 86 L185 90" stroke="#a78bfa" stroke-width="1.5"/>
  <path d="M250 60 Q270 40 290 60 Q310 80 290 95 Q270 110 250 95 Q230 80 250 60Z" fill="none" stroke="#ef4444" stroke-width="2.5"/>
  <path d="M260 77 L270 67 L275 75 L280 60 L285 85 L290 70 L295 77" stroke="#ef4444" stroke-width="2" fill="none"/>
  <text x="270" y="125" text-anchor="middle" fill="#f472b6" font-size="10" font-family="monospace">PLEURAL EFFUSION</text>
  <text x="200" y="168" text-anchor="middle" fill="#a78bfa" font-size="13" font-family="monospace" font-weight="bold">HCM — DYSPNOEIC CAT</text>
  <path d="M175 105 Q185 90 195 105 Q200 95 210 105" stroke="#f59e0b" stroke-width="1.5" fill="none" stroke-dasharray="3"/>
  <text x="195" y="100" fill="#f59e0b" font-size="9" font-family="monospace">laboured</text>
</svg>""",

    "hard_imha_budget": """<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="180" fill="#0d1b2a"/>
  <ellipse cx="180" cy="115" rx="80" ry="48" fill="#1c1010" stroke="#dc2626" stroke-width="2"/>
  <circle cx="130" cy="78" r="32" fill="#1c1010" stroke="#dc2626" stroke-width="2"/>
  <ellipse cx="130" cy="65" rx="13" ry="9" fill="#2d1515"/>
  <ellipse cx="121" cy="60" rx="5" ry="7" fill="#1c1010" stroke="#dc2626" stroke-width="1.5"/>
  <ellipse cx="139" cy="60" rx="5" ry="7" fill="#1c1010" stroke="#dc2626" stroke-width="1.5"/>
  <circle cx="121" cy="78" r="4" fill="#fbbf24" opacity="0.8"/>
  <circle cx="139" cy="78" r="4" fill="#fbbf24" opacity="0.8"/>
  <path d="M115 88 Q130 94 145 88" stroke="#fbbf24" stroke-width="2" fill="none"/>
  <circle cx="310" cy="75" r="35" fill="#1c1c10" stroke="#fbbf24" stroke-width="2"/>
  <text x="310" y="65" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="monospace" font-weight="bold">₹38,000</text>
  <text x="310" y="78" text-anchor="middle" fill="#ef4444" font-size="10" font-family="monospace">BUDGET</text>
  <rect x="285" y="88" width="50" height="8" rx="4" fill="#1f2937"/>
  <rect x="285" y="88" width="18" height="8" rx="4" fill="#ef4444"/>
  <text x="310" y="108" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="monospace">35% used</text>
  <text x="200" y="168" text-anchor="middle" fill="#dc2626" font-size="13" font-family="monospace" font-weight="bold">IMHA — PALE/ICTERIC</text>
  <circle cx="230" cy="90" r="6" fill="#fbbf24" opacity="0.6"/>
  <circle cx="245" cy="100" r="4" fill="#fbbf24" opacity="0.4"/>
  <circle cx="220" cy="108" r="5" fill="#fbbf24" opacity="0.5"/>
</svg>""",

    "hard_polytrauma": """<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="180" fill="#0d1b2a"/>
  <rect x="0" y="0" width="400" height="180" fill="url(#trauma-bg)" opacity="0.3"/>
  <ellipse cx="185" cy="118" rx="85" ry="46" fill="#1a1a2e" stroke="#f59e0b" stroke-width="2"/>
  <circle cx="130" cy="78" r="30" fill="#1a1a2e" stroke="#f59e0b" stroke-width="2"/>
  <ellipse cx="130" cy="65" rx="12" ry="8" fill="#2a2a1e"/>
  <ellipse cx="121" cy="60" rx="5" ry="7" fill="#1a1a2e" stroke="#f59e0b" stroke-width="1.5"/>
  <ellipse cx="139" cy="60" rx="5" ry="7" fill="#1a1a2e" stroke="#f59e0b" stroke-width="1.5"/>
  <circle cx="121" cy="78" r="3" fill="#fcd34d"/>
  <circle cx="139" cy="78" r="3" fill="#fcd34d"/>
  <line x1="100" y1="70" x2="80" y2="55" stroke="#ef4444" stroke-width="3"/>
  <line x1="105" y1="75" x2="82" y2="68" stroke="#ef4444" stroke-width="2"/>
  <rect x="270" y="50" width="60" height="80" rx="4" fill="#0f172a" stroke="#f59e0b" stroke-width="1.5"/>
  <rect x="278" y="58" width="44" height="55" rx="2" fill="#1e293b"/>
  <path d="M285 85 Q300 65 315 85" stroke="#ef4444" stroke-width="2" fill="none"/>
  <path d="M285 100 Q295 88 305 95 Q315 100 318 90" stroke="#ef4444" stroke-width="1.5" fill="none"/>
  <text x="300" y="122" text-anchor="middle" fill="#f59e0b" font-size="8" font-family="monospace">RADIOGRAPH</text>
  <polygon points="355,30 365,50 345,50" fill="#ef4444" opacity="0.9"/>
  <text x="355" y="46" text-anchor="middle" fill="white" font-size="10" font-weight="bold">!</text>
  <text x="200" y="168" text-anchor="middle" fill="#f59e0b" font-size="13" font-family="monospace" font-weight="bold">POLYTRAUMA — HBC</text>
</svg>""",

    "hard_stochastic_pancreatitis": """<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="180" fill="#0d1b2a"/>
  <ellipse cx="175" cy="118" rx="80" ry="44" fill="#1a1f1a" stroke="#22c55e" stroke-width="2"/>
  <circle cx="120" cy="78" r="30" fill="#1a1f1a" stroke="#22c55e" stroke-width="2"/>
  <ellipse cx="120" cy="65" rx="14" ry="9" fill="#1e2a1e"/>
  <ellipse cx="110" cy="60" rx="5" ry="8" fill="#1a1f1a" stroke="#22c55e" stroke-width="1.5"/>
  <ellipse cx="130" cy="60" rx="5" ry="8" fill="#1a1f1a" stroke="#22c55e" stroke-width="1.5"/>
  <circle cx="112" cy="78" r="4" fill="#4ade80"/>
  <circle cx="128" cy="78" r="4" fill="#4ade80"/>
  <path d="M105 88 Q120 82 135 88" stroke="#ef4444" stroke-width="2" fill="none"/>
  <text x="290" y="55" text-anchor="middle" fill="#ef4444" font-size="28" font-family="monospace" font-weight="bold">⚡</text>
  <rect x="250" y="68" width="80" height="18" rx="4" fill="#1e293b" stroke="#ef4444" stroke-width="1"/>
  <text x="290" y="81" text-anchor="middle" fill="#f87171" font-size="9" font-family="monospace">ACTION FAILED</text>
  <rect x="250" y="92" width="80" height="18" rx="4" fill="#1e293b" stroke="#f59e0b" stroke-width="1"/>
  <text x="290" y="105" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="monospace">RETRY DIFF ROUTE</text>
  <rect x="250" y="116" width="80" height="18" rx="4" fill="#1e293b" stroke="#22c55e" stroke-width="1"/>
  <text x="290" y="129" text-anchor="middle" fill="#4ade80" font-size="9" font-family="monospace">cooperation: 0.4</text>
  <text x="200" y="168" text-anchor="middle" fill="#22c55e" font-size="13" font-family="monospace" font-weight="bold">PANCREATITIS — STOCHASTIC</text>
</svg>""",

    "hard_parvovirus_day1": """<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="180" fill="#0d1b2a"/>
  <ellipse cx="160" cy="125" rx="65" ry="38" fill="#1a1020" stroke="#d946ef" stroke-width="2"/>
  <circle cx="115" cy="88" r="26" fill="#1a1020" stroke="#d946ef" stroke-width="2"/>
  <ellipse cx="115" cy="77" rx="11" ry="7" fill="#251030"/>
  <ellipse cx="107" cy="73" rx="4" ry="6" fill="#1a1020" stroke="#d946ef" stroke-width="1.5"/>
  <ellipse cx="123" cy="73" rx="4" ry="6" fill="#1a1020" stroke="#d946ef" stroke-width="1.5"/>
  <circle cx="108" cy="88" r="3" fill="#e879f9"/>
  <circle cx="122" cy="88" r="3" fill="#e879f9"/>
  <rect x="250" y="45" width="95" height="100" rx="8" fill="#0f172a" stroke="#d946ef" stroke-width="2"/>
  <rect x="260" y="55" width="75" height="30" rx="4" fill="#1e293b"/>
  <text x="297" y="68" text-anchor="middle" fill="#a855f7" font-size="9" font-family="monospace">SNAP PARVO</text>
  <text x="297" y="80" text-anchor="middle" fill="#ef4444" font-size="14" font-family="monospace" font-weight="bold">NEGATIVE</text>
  <text x="297" y="100" text-anchor="middle" fill="#f59e0b" font-size="8" font-family="monospace">⚠ 25% FALSE NEG</text>
  <text x="297" y="112" text-anchor="middle" fill="#94a3b8" font-size="8" font-family="monospace">on day 1 only</text>
  <rect x="260" y="120" width="75" height="16" rx="3" fill="#7f1d1d"/>
  <text x="297" y="131" text-anchor="middle" fill="#fca5a5" font-size="8" font-family="monospace">TREAT ANYWAY →</text>
  <text x="200" y="168" text-anchor="middle" fill="#d946ef" font-size="13" font-family="monospace" font-weight="bold">PARVOVIRUS DAY 1</text>
</svg>""",

    "hard_nosocomial_chf_ward": """<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="180" fill="#0d1b2a"/>
  <rect x="60" y="100" width="160" height="55" rx="4" fill="#0f172a" stroke="#0891b2" stroke-width="2"/>
  <rect x="70" y="110" width="140" height="35" rx="2" fill="#1e293b"/>
  <ellipse cx="140" cy="100" rx="55" ry="30" fill="#0f172a" stroke="#0891b2" stroke-width="2"/>
  <circle cx="108" cy="85" r="22" fill="#0f172a" stroke="#0891b2" stroke-width="1.5"/>
  <circle cx="100" cy="80" r="3" fill="#7dd3fc"/>
  <circle cx="116" cy="80" r="3" fill="#7dd3fc"/>
  <line x1="60" y1="80" x2="60" y2="30" stroke="#0891b2" stroke-width="2"/>
  <rect x="50" y="25" width="20" height="30" rx="3" fill="#0e7490" stroke="#0891b2" stroke-width="1.5"/>
  <line x1="60" y1="55" x2="75" y2="90" stroke="#0891b2" stroke-width="1.5" stroke-dasharray="4"/>
  <rect x="270" y="30" width="90" height="120" rx="6" fill="#0f172a" stroke="#0891b2" stroke-width="1.5"/>
  <text x="315" y="50" text-anchor="middle" fill="#7dd3fc" font-size="9" font-family="monospace">INFECTION RISK</text>
  <rect x="280" y="55" width="70" height="12" rx="3" fill="#064e3b"/>
  <text x="315" y="65" text-anchor="middle" fill="#6ee7b7" font-size="8" font-family="monospace">24h: 10% ✓</text>
  <rect x="280" y="72" width="70" height="12" rx="3" fill="#78350f"/>
  <text x="315" y="82" text-anchor="middle" fill="#fcd34d" font-size="8" font-family="monospace">48h: 25% ⚠</text>
  <rect x="280" y="89" width="70" height="12" rx="3" fill="#7f1d1d"/>
  <text x="315" y="99" text-anchor="middle" fill="#fca5a5" font-size="8" font-family="monospace">72h: 50% ✗</text>
  <rect x="280" y="106" width="70" height="12" rx="3" fill="#450a0a"/>
  <text x="315" y="116" text-anchor="middle" fill="#ef4444" font-size="8" font-family="monospace">96h: 75% ✗✗</text>
  <text x="315" y="136" text-anchor="middle" fill="#f59e0b" font-size="9" font-family="monospace">DISCHARGE</text>
  <text x="315" y="148" text-anchor="middle" fill="#f59e0b" font-size="9" font-family="monospace">ASAP ↓</text>
  <text x="170" y="168" text-anchor="middle" fill="#0891b2" font-size="13" font-family="monospace" font-weight="bold">CHF WARD — NOSOCOMIAL</text>
</svg>""",
}

TASK_META = {
    "easy_gdv": {
        "emoji": "🐕", "label": "GDV Emergency", "badge": "🟢 EASY",
        "desc": "German Shepherd with acute gastric dilatation-volvulus. Distended abdomen, unproductive retching.",
        "tip": "💡 Check cardiovascular vitals → abdominal exam → immediate_resuscitation → admit_icu. NEVER discharge a GDV.",
    },
    "medium_hcm_cat": {
        "emoji": "🐈", "label": "HCM Cat", "badge": "🟡 MEDIUM",
        "desc": "Dyspnoeic cat with hypertrophic cardiomyopathy. Rapid laboured breathing, pleural effusion suspected.",
        "tip": "💡 Do NOT give crystalloid — causes pulmonary oedema. Use furosemide IV. Thoracocentesis if effusion confirmed.",
    },
    "hard_imha_budget": {
        "emoji": "🩸", "label": "IMHA Budget", "badge": "🔴 HARD",
        "desc": "Immune haemolytic anaemia — pale/icteric mucous membranes, weakness, tachycardia. Hard ₹38,000 budget.",
        "tip": "💡 Contact owner FIRST to learn budget. CBC + lactate only. Blood products over crystalloid. Skip imaging.",
    },
    "hard_polytrauma": {
        "emoji": "🚑", "label": "Polytrauma HBC", "badge": "🔴 HARD",
        "desc": "Hit-by-car dog. Multiple injuries — pneumothorax, haemoabdomen, fractures. Mid-episode seizure.",
        "tip": "💡 Image thorax + abdomen URGENTLY. Prefer colloid over crystalloid. Prepare for seizure at step ~8.",
    },
    "hard_stochastic_pancreatitis": {
        "emoji": "⚡", "label": "Stochastic Pancreatitis", "badge": "🔴 HARD",
        "desc": "Stress-reactive Border Collie with severe pancreatitis. Cooperation 0.4 — actions fail silently 2.5× more.",
        "tip": "💡 Check action_succeeded after EVERY step. If tool failed silently, switch route/method/site immediately.",
    },
    "hard_parvovirus_day1": {
        "emoji": "🦠", "label": "Parvo Day 1", "badge": "🔴 HARD",
        "desc": "Unvaccinated puppy: haemorrhagic diarrhoea, vomiting, shock. SNAP test 25% false-negative on day 1.",
        "tip": "💡 Negative SNAP ≠ no parvo on day 1. Leukopenia + bloody diarrhoea = treat empirically regardless.",
    },
    "hard_nosocomial_chf_ward": {
        "emoji": "🏥", "label": "Nosocomial CHF", "badge": "🔴 HARD",
        "desc": "CHF dog 20h into ward stay — partially stabilised. Hospital-acquired infection risk escalates every 24h.",
        "tip": "💡 Patient is stable — discharge ASAP. Risk: 10%@24h → 25%@48h → 50%@72h. Don't over-monitor.",
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
    svg = TASK_SVGS.get(task_id, "")
    img_html = f'<div style="width:100%;margin-bottom:4px">{svg}</div>' if svg else ""
    card = _task_card(task_id)
    return img_html, card


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

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI veterinary triage agent. Assess and treat animal patients efficiently.
    Each turn respond with ONLY a valid JSON object:
    {"tool": "<tool_name>", "parameters": {<params>}, "reasoning": "<brief reason>"}

    Tools: check_vitals, physical_exam, run_bloodwork, run_imaging, collect_result,
    place_iv_access, administer_fluid_bolus, give_medication, oxygen_therapy,
    perform_procedure, contact_owner, consult_specialist, decide_triage_route, make_disposition.

    Rules:
    - Check action_succeeded after every step. If False, adapt (change route/site/method).
    - GDV: immediate_resuscitation → admit_icu. NEVER discharge.
    - HCM cat: NO crystalloid. Use furosemide.
    - Negative SNAP parvo on day 1 = treat anyway on clinical signs.
    - Discharge CHF ward patient ASAP to avoid nosocomial infection.
""").strip()


def run_llm_episode(task_id: str, base_url: str, model: str, token: str) -> str:
    """Run a full episode with an LLM and return HTML summary."""
    try:
        from openai import OpenAI
    except ImportError:
        return '<div style="color:#f87171">openai package not installed</div>'

    api_key = token.strip() or os.getenv("HF_TOKEN") or ""
    if not api_key:
        return '<div style="color:#f87171">❌ No HF Token provided. Add it above or set HF_TOKEN Space secret.</div>'

    client = OpenAI(base_url=base_url.strip(), api_key=api_key)

    env = VetTriageEnv(max_total_steps=100)
    obs = env.reset(task_id=task_id, seed=42)

    history = []
    rewards = []
    log_rows = []
    score = 0.0
    passed = False

    for step in range(1, 41):
        if obs.episode_done:
            break

        # Build prompt
        action_warn = ""
        if not obs.action_succeeded:
            action_warn = f"\n⚠ LAST ACTION FAILED: {obs.latest_clinical_event} — ADAPT NOW!"

        user_msg = (
            f"Phase: {obs.phase} | Step: {obs.step}/{obs.phase_step_limit} | Time: {obs.sim_time_hours:.1f}h\n"
            f"Complaint: {obs.presenting_complaint[:120]}\n"
            f"Vitals: {json.dumps(obs.vitals.model_dump() if obs.vitals and hasattr(obs.vitals,'model_dump') else obs.vitals or 'not checked')}\n"
            f"Exam: {obs.physical_exam_findings}\n"
            f"Labs: {obs.lab_results}\n"
            f"Events: {obs.events}\n"
            f"Available: {obs.available_tools}{action_warn}\n"
            f"History (last 4): {history[-4:]}\n"
            "Choose next action as JSON."
        )

        tool, params, reasoning = "check_vitals", {"systems": ["cardiovascular"]}, "fallback"
        llm_error = None
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1, max_tokens=200,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text.strip())
            tool = parsed["tool"]
            params = parsed.get("parameters", {})
            reasoning = parsed.get("reasoning", "")
        except Exception as e:
            err_str = str(e)
            # Detect quota exhaustion / billing errors — stop immediately
            if any(k in err_str for k in ("402", "depleted", "quota", "billing", "credit", "payment")):
                llm_error = err_str
            else:
                reasoning = f"parse error: {e}"

        if llm_error:
            return f"""
            <div style="background:#450a0a;border:1px solid #991b1b;border-radius:12px;padding:20px;text-align:center">
              <div style="font-size:1.3rem;font-weight:800;color:#fca5a5;margin-bottom:8px">💳 LLM Quota Exhausted</div>
              <div style="color:#f87171;font-size:0.88rem;margin-bottom:12px">
                The model endpoint returned a quota/billing error after {len(rewards)} steps.
              </div>
              <div style="background:#1c0505;border-radius:8px;padding:10px;font-family:monospace;font-size:0.78rem;color:#fca5a5;text-align:left;word-break:break-all">
                {llm_error[:400]}
              </div>
              <div style="margin-top:12px;color:#94a3b8;font-size:0.82rem">
                💡 Use a paid endpoint, switch to a free model (e.g. mistralai/Mistral-7B-Instruct-v0.3),
                or set a different API_BASE_URL.
              </div>
            </div>
            """

        action = Action(tool=tool, parameters=params, reasoning=reasoning)
        obs, reward, done, info = env.step(action)
        rewards.append(reward.value)
        history.append(f"Step {step}: {tool} → {reward.value:+.2f}")

        fail_tag = " ❌" if not obs.action_succeeded else ""
        r_colour = "#34d399" if reward.value >= 0 else "#f87171"
        log_rows.append(
            f'<tr>'
            f'<td style="padding:4px 8px;color:#94a3b8">{step}</td>'
            f'<td style="padding:4px 8px;color:#a5f3fc;font-family:monospace">{tool}{fail_tag}</td>'
            f'<td style="padding:4px 8px;color:{r_colour};font-weight:700">{reward.value:+.3f}</td>'
            f'<td style="padding:4px 8px;color:#64748b;font-size:0.78rem">{reasoning[:60]}</td>'
            f'</tr>'
        )

        if done:
            score = info.get("grade", 0.0)
            passed = info.get("passed", False)
            break

    grade_colour = "#34d399" if score >= 0.7 else "#f59e0b" if score >= 0.5 else "#f87171"
    pass_badge = (
        '<span style="background:#064e3b;color:#6ee7b7;border-radius:6px;padding:2px 8px;font-weight:700">✅ PASSED</span>'
        if passed else
        '<span style="background:#7f1d1d;color:#fca5a5;border-radius:6px;padding:2px 8px;font-weight:700">❌ FAILED</span>'
    )

    return f"""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
        <div style="font-size:1.8rem;font-weight:900;color:{grade_colour}">{score:.3f}</div>
        <div>
          <div style="color:#94a3b8;font-size:0.8rem">Final Score · {len(rewards)} steps · {model.split('/')[-1]}</div>
          <div style="margin-top:4px">{pass_badge}</div>
        </div>
      </div>
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
          <thead><tr style="border-bottom:1px solid #1f2937">
            <th style="padding:4px 8px;color:#60a5fa;text-align:left">#</th>
            <th style="padding:4px 8px;color:#60a5fa;text-align:left">Tool</th>
            <th style="padding:4px 8px;color:#60a5fa;text-align:left">Reward</th>
            <th style="padding:4px 8px;color:#60a5fa;text-align:left">Reasoning</th>
          </tr></thead>
          <tbody>{"".join(log_rows)}</tbody>
        </table>
      </div>
    </div>
    """


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

                task_img = gr.HTML(
                    f'<div style="width:100%;margin-bottom:4px">{TASK_SVGS.get(default_task, "")}</div>'
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

        # ── LLM Auto-Play ──
        gr.HTML('<hr style="border-color:#1f2937;margin:16px 0">')
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#1e1b4b,#0f172a);border:1px solid #312e81;
             border-radius:12px;padding:16px;margin-bottom:8px">
          <div style="color:#a5b4fc;font-size:0.75rem;font-weight:700;text-transform:uppercase;
               letter-spacing:0.08em;margin-bottom:6px">🤖 LLM Auto-Play</div>
          <div style="color:#94a3b8;font-size:0.83rem">
            Run a full episode automatically using an LLM via the OpenAI-compatible API.
            Requires <code style="color:#a5f3fc">API_BASE_URL</code>,
            <code style="color:#a5f3fc">MODEL_NAME</code>, and
            <code style="color:#a5f3fc">HF_TOKEN</code> set as Space secrets.
          </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                llm_base_url = gr.Textbox(
                    value=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
                    label="API Base URL",
                )
                llm_model = gr.Textbox(
                    value=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
                    label="Model Name",
                )
                llm_token = gr.Textbox(
                    value="",
                    label="HF Token (leave blank to use Space secret)",
                    type="password",
                )
            with gr.Column(scale=1):
                autoplay_btn = gr.Button("🤖  Run LLM Episode", size="lg",
                    elem_id="start-btn", interactive=True)
                autoplay_status = gr.HTML(
                    '<div style="color:#475569;font-style:italic;font-size:0.83rem">'
                    'LLM results will appear here.</div>'
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

        autoplay_btn.click(
            fn=lambda task, url, model, tok: run_llm_episode(task, url, model, tok),
            inputs=[task_dd, llm_base_url, llm_model, llm_token],
            outputs=[autoplay_status],
        )

    return demo

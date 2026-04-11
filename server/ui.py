"""
VetTriageEnv — Gradio UI
Mounted at /ui on the FastAPI server.
"""
from __future__ import annotations

import json
import textwrap
from typing import Optional

import gradio as gr

from vettriagevenv.env import VetTriageEnv
from vettriagevenv.models import Action
from vettriagevenv.tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Shared env instance (one session at a time — demo mode)
# ---------------------------------------------------------------------------
_env = VetTriageEnv(max_total_steps=100)
_state: dict = {}   # holds last obs + metadata between steps


TASK_CHOICES = list(TASK_REGISTRY.keys())

TOOL_CHOICES = [
    "check_vitals",
    "physical_exam",
    "run_bloodwork",
    "run_imaging",
    "collect_result",
    "place_iv_access",
    "administer_fluid_bolus",
    "give_medication",
    "oxygen_therapy",
    "perform_procedure",
    "contact_owner",
    "consult_specialist",
    "decide_triage_route",
    "make_disposition",
]

TOOL_PARAM_HINTS = {
    "check_vitals":          '{"systems": ["cardiovascular", "respiratory", "temperature", "pain", "mentation"]}',
    "physical_exam":         '{"region": "abdomen"}',
    "run_bloodwork":         '{"panel": "cbc"}',
    "run_imaging":           '{"modality": "radiograph", "region": "thorax"}',
    "collect_result":        '{"job_id": "<from pending results>"}',
    "place_iv_access":       '{"site": "cephalic"}',
    "administer_fluid_bolus":'{"fluid_type": "crystalloid", "dose_ml_kg": 10}',
    "give_medication":       '{"drug": "morphine", "dose": "0.2mg/kg", "route": "iv"}',
    "oxygen_therapy":        '{"method": "mask"}',
    "perform_procedure":     '{"procedure": "gastric_decompression"}',
    "contact_owner":         '{"message": "Your pet needs urgent treatment."}',
    "consult_specialist":    '{"specialty": "surgery", "question": "Is GDV confirmed?"}',
    "decide_triage_route":   '{"route": "immediate_resuscitation"}',
    "make_disposition":      '{"disposition": "admit_icu"}',
}

TASK_DESCRIPTIONS = {
    "easy_gdv":                    "🟢 Easy — GDV emergency dog. Clear signals, baseline test.",
    "medium_hcm_cat":              "🟡 Medium — HCM dyspnoeic cat. Wait-vs-treat time dilemma.",
    "hard_imha_budget":            "🔴 Hard — IMHA with ₹38,000 hard budget constraint.",
    "hard_polytrauma":             "🔴 Hard — Hit-by-car polytrauma. Mid-episode seizure.",
    "hard_stochastic_pancreatitis":"🔴 Hard — Uncooperative patient. Stochastic action failures.",
    "hard_parvovirus_day1":        "🔴 Hard — False-negative SNAP parvo test. Clinical gestalt required.",
    "hard_nosocomial_chf_ward":    "🔴 Hard — CHF ward stay. Escalating hospital-infection clock.",
}


def _fmt_obs(obs) -> str:
    """Format observation into readable markdown."""
    lines = []
    lines.append(f"**Phase:** `{obs.phase}` | **Step:** {obs.step}/{obs.phase_step_limit} | **Sim time:** {obs.sim_time_hours:.1f}h")

    if obs.budget_limit is not None:
        remaining = obs.budget_remaining if obs.budget_remaining is not None else obs.budget_limit - obs.budget_spent
        lines.append(f"**Budget:** ₹{remaining:.0f} remaining of ₹{obs.budget_limit:.0f}")

    if not obs.action_succeeded:
        lines.append(f"\n⚠️ **LAST ACTION FAILED:** {obs.latest_clinical_event}")
    elif obs.latest_clinical_event:
        lines.append(f"*Clinical event:* {obs.latest_clinical_event}")

    if obs.vitals:
        v = obs.vitals if isinstance(obs.vitals, dict) else obs.vitals.model_dump()
        lines.append("\n**Vitals:**")
        for k, val in v.items():
            if val is not None and k != "systems_checked":
                lines.append(f"  - {k.replace('_', ' ').title()}: `{val}`")

    if obs.physical_exam_findings:
        lines.append("\n**Physical Exam:**")
        for region, finding in obs.physical_exam_findings.items():
            lines.append(f"  - {region}: {finding}")

    if obs.lab_results:
        lines.append("\n**Lab Results:**")
        for panel, result in obs.lab_results.items():
            if isinstance(result, dict):
                interp = result.get("interpretation", "")
                lines.append(f"  - {panel}: {interp}")
            else:
                lines.append(f"  - {panel}: {result}")

    if obs.imaging_results:
        lines.append("\n**Imaging:**")
        for key, result in obs.imaging_results.items():
            if isinstance(result, dict):
                interp = result.get("interpretation", "")
                lines.append(f"  - {key}: {interp}")

    if obs.pending_results:
        lines.append("\n**Pending Results:**")
        for p in obs.pending_results:
            if isinstance(p, dict):
                lines.append(f"  - `{p.get('job_id')}` — {p.get('panel_or_modality')} (ready in ~{p.get('eta_steps')} steps)")
            else:
                lines.append(f"  - `{p.job_id}` — {p.panel_or_modality} (ready in ~{p.eta_steps} steps)")

    if obs.specialist_opinion:
        lines.append(f"\n**Specialist Opinion:** {obs.specialist_opinion}")

    if obs.monitoring_trends:
        lines.append("\n**Monitoring Trends:**")
        for k, v in obs.monitoring_trends.items():
            lines.append(f"  - {k}: {v}")

    if obs.events:
        lines.append("\n**🚨 Events:**")
        for e in obs.events:
            lines.append(f"  > {e}")

    if obs.available_tools:
        lines.append(f"\n**Available tools:** {', '.join(f'`{t}`' for t in obs.available_tools)}")

    return "\n".join(lines)


def _fmt_history(obs) -> str:
    if not obs.action_history:
        return "*No actions taken yet.*"
    return "\n".join(f"`{h}`" for h in obs.action_history[-10:])


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------

def do_reset(task_id: str):
    global _state
    obs = _env.reset(task_id=task_id, seed=42)
    _state = {"obs": obs, "done": False, "score": None, "rewards": [], "step": 0}
    patient_info = (
        f"**Patient:** {obs.species.title()} | {obs.breed} | {obs.age_years:.1f}yr | "
        f"{obs.sex} | {obs.weight_kg:.1f}kg\n\n"
        f"**Presenting Complaint:**\n{obs.presenting_complaint}"
    )
    return (
        patient_info,
        _fmt_obs(obs),
        _fmt_history(obs),
        "Episode started. Choose a tool and take your first action.",
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def update_params(tool: str):
    return TOOL_PARAM_HINTS.get(tool, "{}")


def do_step(tool: str, params_str: str, reasoning: str):
    global _state
    if not _state or _state.get("done"):
        return (
            _state.get("obs_md", ""),
            _state.get("hist_md", ""),
            "⚠️ No active episode. Click **Start Episode** first.",
            gr.update(interactive=False),
        )

    try:
        params = json.loads(params_str) if params_str.strip() else {}
    except json.JSONDecodeError as e:
        return (
            _fmt_obs(_state["obs"]),
            _fmt_history(_state["obs"]),
            f"❌ Invalid JSON in parameters: {e}",
            gr.update(interactive=True),
        )

    action = Action(tool=tool, parameters=params, reasoning=reasoning or None)
    obs, reward, done, info = _env.step(action)
    _state["obs"] = obs
    _state["rewards"].append(reward.value)
    _state["step"] += 1

    status_parts = [f"**Step {_state['step']}** | Tool: `{tool}` | Reward: `{reward.value:+.3f}`"]
    if reward.message:
        status_parts.append(f"*{reward.message}*")

    if done:
        _state["done"] = True
        grade = info.get("grade", 0.0)
        passed = info.get("passed", False)
        feedback = info.get("grade_feedback", [])
        _state["score"] = grade
        status_parts.append(f"\n---\n### Episode Complete!")
        status_parts.append(f"**Final Score:** `{grade:.3f}` | **Passed:** {'✅' if passed else '❌'}")
        if feedback:
            status_parts.append("\n**Feedback:**")
            for f in feedback:
                status_parts.append(f"  - {f}")
        return (
            _fmt_obs(obs),
            _fmt_history(obs),
            "\n".join(status_parts),
            gr.update(interactive=False),
        )

    return (
        _fmt_obs(obs),
        _fmt_history(obs),
        "\n".join(status_parts),
        gr.update(interactive=True),
    )


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="VetTriageEnv",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css="""
        .patient-box { background: #1e293b; border-radius: 8px; padding: 12px; }
        .status-box  { background: #0f172a; border-radius: 8px; padding: 12px; }
        """,
    ) as demo:

        gr.Markdown(
            """
            # 🐾 VetTriageEnv — Interactive Demo
            An AI veterinary triage environment. Gather clinical information via tool calls and make the correct triage decision.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                task_dd = gr.Dropdown(
                    choices=TASK_CHOICES,
                    value="easy_gdv",
                    label="Select Task",
                    info="Choose a benchmark task",
                )
                task_desc = gr.Markdown(TASK_DESCRIPTIONS["easy_gdv"])
                start_btn = gr.Button("▶ Start Episode", variant="primary", size="lg")
                gr.Markdown("---")
                patient_md = gr.Markdown("*Start an episode to see patient info.*", label="Patient")

            with gr.Column(scale=2):
                obs_md = gr.Markdown("*Start an episode to see the observation.*", label="Observation")

        gr.Markdown("---")
        gr.Markdown("### Take an Action")

        with gr.Row():
            with gr.Column(scale=1):
                tool_dd = gr.Dropdown(
                    choices=TOOL_CHOICES,
                    value="check_vitals",
                    label="Tool",
                )
                params_box = gr.Textbox(
                    value=TOOL_PARAM_HINTS["check_vitals"],
                    label="Parameters (JSON)",
                    lines=3,
                )
                reasoning_box = gr.Textbox(
                    placeholder="Optional: explain your reasoning",
                    label="Reasoning",
                    lines=2,
                )
                step_btn = gr.Button("⚡ Take Action", variant="primary", interactive=False)

            with gr.Column(scale=1):
                status_md = gr.Markdown("*Status will appear here.*", label="Step Result")
                history_md = gr.Markdown("*Action history will appear here.*", label="Action History (last 10)")

        # --- wiring ---
        task_dd.change(
            fn=lambda t: TASK_DESCRIPTIONS.get(t, ""),
            inputs=task_dd,
            outputs=task_desc,
        )

        tool_dd.change(
            fn=update_params,
            inputs=tool_dd,
            outputs=params_box,
        )

        start_btn.click(
            fn=do_reset,
            inputs=task_dd,
            outputs=[patient_md, obs_md, history_md, status_md, step_btn, step_btn],
        )

        step_btn.click(
            fn=do_step,
            inputs=[tool_dd, params_box, reasoning_box],
            outputs=[obs_md, history_md, status_md, step_btn],
        )

    return demo

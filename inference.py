#!/usr/bin/env python3
"""
VetTriageEnv — OpenEnv Hackathon inference script.

Mandatory env vars:
    API_BASE_URL   LLM endpoint  (default: HF router)
    MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key
    ENV_URL        HF Space URL  (default: https://vantatess-vettriagevenv.hf.space)
"""
import json
import os
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
ENV_URL      = os.getenv("ENV_URL", "https://vantatess-vettriagevenv.hf.space")
BENCHMARK    = "vettriagevenv"
MAX_STEPS    = 90  # env allows 100 total; leave 10 buffer for terminal actions

TASKS = [
    "easy_gdv",
    "medium_hcm_cat",
    "hard_imha_budget",
    "hard_polytrauma",
    "hard_stochastic_pancreatitis",
    "hard_parvovirus_day1",
    "hard_nosocomial_chf_ward",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI veterinary triage agent. Assess and treat animal patients efficiently.

    Tools you may call (one per turn):
      check_vitals        {"systems": ["cardiovascular","respiratory","temperature","pain","mentation"]}
      physical_exam       {"region": "abdomen"|"thorax"|"musculoskeletal"|"neurological"|"skin"}
      run_bloodwork       {"panel": "cbc"|"chemistry"|"coagulation"|"lactate"}
      run_imaging         {"modality": "radiograph"|"ultrasound", "region": "thorax"|"abdomen"|"spine"}
      collect_result      {"job_id": "<id from pending_results>"}
      place_iv_access     {"site": "cephalic"|"saphenous"|"jugular"}
      administer_fluid_bolus {"fluid_type": "crystalloid"|"colloid", "dose_ml_kg": <number>}
      give_medication     {"drug": "<name>", "dose": "<dose>", "route": "iv"|"im"|"po"}
      oxygen_therapy      {"method": "flow-by"|"mask"|"nasal_cannula"|"cage"}
      perform_procedure   {"procedure": "thoracocentesis"|"gastric_decompression"|"urinary_catheter"}
      contact_owner       {"message": "<msg>"}
      consult_specialist  {"specialty": "cardiology"|"surgery"|"neurology", "question": "<q>"}
      decide_triage_route {"route": "immediate_resuscitation"|"urgent_stabilise"|"standard_triage"|"monitor_observe"}
      make_disposition    {"disposition": "admit_icu"|"admit_ward"|"discharge_with_medication"|"discharge_home"|"refer_specialist"|"euthanasia"}

    CRITICAL RULES — ALWAYS follow these:

    1. STOCHASTIC FAILURES: After EVERY action, check action_succeeded and latest_clinical_event.
       - If action_succeeded=false, the intervention DID NOT work. Adapt immediately:
         * Oxygen mask failed → switch to {"method": "cage"} or {"method": "nasal_cannula"}
         * IV placement failed → retry with different site (saphenous if cephalic failed)
         * Oral medication failed → switch route to {"route": "iv"}
       - Silent failures look like success — action_succeeded is the only reliable indicator.

    2. BUDGET: Always call contact_owner first on budget-constrained tasks to learn the limit.
       - Check budget_remaining before ordering expensive diagnostics.
       - CBC+lactate (cheap) beats full_panel+ultrasound (expensive) when budget is tight.

    3. TIME PRESSURE: Treat first, diagnose later when SpO2 is falling or patient is critical.
       - Oxygen therapy and stabilisation come before confirmatory tests.
       - For HCM cats: DO NOT give crystalloid bolus (causes pulmonary oedema). Use furosemide.
       - For haemoabdomen: prefer colloid over crystalloid.

    4. FALSE NEGATIVES: A negative SNAP parvo test in an unvaccinated puppy with haemorrhagic
       diarrhoea does NOT rule out parvo. Clinical signs override a day-1 test. Treat anyway.

    5. NOSOCOMIAL HAZARD: Check monitoring_trends["nosocomial_risk"] each step.
       - Discharge the patient before the 24h infection window closes.
       - Do NOT over-monitor (repeat bloodwork, unnecessary consults) if patient is stable.

    6. PHASE SEQUENCE: triage → stabilisation → monitoring → disposition.
       - Triage: check vitals + key exam → decide_triage_route.
       - GDV dogs: immediate_resuscitation → admit_icu. NEVER discharge a GDV.
       - Stabilisation: IV access → fluids/oxygen/medications/procedures.
       - Monitoring: collect pending results, check trends, plan discharge.
       - Disposition: make_disposition when patient is stable.

    Respond with ONLY a valid JSON object (no markdown, no explanation):
    {"tool": "<tool_name>", "parameters": {<params>}, "reasoning": "<brief reason>"}
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers (exact format required by hackathon)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def reset_env(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def step_env(tool: str, parameters: dict, reasoning: str = "") -> dict:
    r = requests.post(
        f"{ENV_URL}/step",
        json={"action": {"tool": tool, "parameters": parameters, "reasoning": reasoning}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

def get_action(client: OpenAI, obs: dict, history: List[str]) -> tuple:
    history_block = "\n".join(history[-6:]) if history else "None"

    # Build stochastic failure warning if last action failed
    action_status = ""
    if not obs.get("action_succeeded", True):
        action_status = f"\n⚠ LAST ACTION FAILED: {obs.get('latest_clinical_event','unknown failure')} — MUST ADAPT!"
    elif obs.get("latest_clinical_event"):
        action_status = f"\nLast clinical event: {obs.get('latest_clinical_event')}"

    # Budget display
    budget_str = "no limit"
    if obs.get("budget_limit") is not None:
        budget_str = f"₹{obs.get('budget_remaining', obs.get('budget_limit', 0)):.0f} remaining of ₹{obs.get('budget_limit'):.0f}"

    user_prompt = textwrap.dedent(f"""
        Patient : {obs.get('species','')} {obs.get('breed','')} {obs.get('age_years','')}yr {obs.get('sex','')} {obs.get('weight_kg','')}kg
        Complaint: {obs.get('presenting_complaint','')}
        Phase    : {obs.get('phase','')}  step {obs.get('step',0)}/{obs.get('phase_step_limit',20)}  sim_time={obs.get('sim_time_hours',0):.1f}h
        Budget   : {budget_str}
        Vitals   : {json.dumps(obs.get('vitals')) if obs.get('vitals') else 'not checked yet'}
        Exam     : {obs.get('physical_exam_findings',{})}
        Labs     : {obs.get('lab_results',{})}
        Imaging  : {obs.get('imaging_results',{})}
        Pending  : {obs.get('pending_results',[])}
        Monitoring: {obs.get('monitoring_trends') or 'N/A'}
        Events   : {obs.get('events',[])}
        Available tools: {obs.get('available_tools',[])}
        {action_status}

        Recent history:
        {history_block}

        Choose your next action as JSON.
    """).strip()

    text = ""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
        return parsed["tool"], parsed.get("parameters", {}), parsed.get("reasoning", "")
    except Exception as e:
        print(f"[DEBUG] LLM parse error: {e} — raw: {text[:200]}", flush=True)
        # Safe fallback
        return (
            "check_vitals",
            {"systems": ["cardiovascular", "respiratory", "temperature", "pain", "mentation"]},
            "fallback after parse error",
        )


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = reset_env(task_id)
        obs = result["observation"]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            # Force a terminal action on the last step so episode always grades
            if step == MAX_STEPS:
                phase = obs.get("phase", "triage")
                if phase == "disposition":
                    tool = "make_disposition"
                    parameters = {"disposition": "admit_icu"}
                    reasoning = "forced terminal: step limit reached"
                else:
                    tool = "decide_triage_route"
                    parameters = {"route": "urgent_stabilise"}
                    reasoning = "forced terminal: step limit reached"
                action_str = f"{tool}({json.dumps(parameters)})"
                error = None
                reward = 0.0
                done = False
                try:
                    step_result = step_env(tool, parameters, reasoning)
                    obs    = step_result["observation"]
                    reward = step_result.get("reward") or 0.0
                    done   = step_result.get("done", False)
                    if done:
                        score   = float(obs.get("grade") or 0.0)
                        success = bool(obs.get("passed") or False)
                except Exception as e:
                    error = str(e)[:120]
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                break

            tool, parameters, reasoning = get_action(client, obs, history)
            action_str = f"{tool}({json.dumps(parameters)})"

            error = None
            reward = 0.0
            done = False

            try:
                step_result = step_env(tool, parameters, reasoning)
                obs    = step_result["observation"]
                reward = step_result.get("reward") or 0.0
                done   = step_result.get("done", False)

                if done:
                    # grade/passed are in observation (metadata is stripped by framework)
                    score   = float(obs.get("grade") or 0.0)
                    success = bool(obs.get("passed") or False)

            except Exception as e:
                error = str(e)[:120]

            rewards.append(reward)
            steps_taken = step
            history.append(f"Step {step}: {tool}({json.dumps(parameters)}) -> reward {reward:+.2f}")

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

    finally:
        # Clamp strictly within (0, 1) — OpenEnv validator rejects 0.0 and 1.0 exactly
        score = max(0.01, min(0.99, score))
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()

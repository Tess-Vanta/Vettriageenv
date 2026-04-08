"""
Rule-based baseline for VetTriageEnv — no API key required.

Uses a deterministic clinical policy for each task and produces
reproducible scores on all benchmark tasks.

Includes two variants of the stochastic pancreatitis policy to demonstrate
the 'Spit-Out' mechanic:
  - open_loop: never reads action_succeeded (silent failures ignored)
  - closed_loop: reads action_succeeded; switches route/method after silent failures

Usage:
    python -X utf8 baseline_rulebased.py
    python -X utf8 baseline_rulebased.py --task easy_gdv
    python -X utf8 baseline_rulebased.py --task hard_stochastic_pancreatitis
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterator

# Force UTF-8 for rupee symbol on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from vettriagevenv import VetTriageEnv, Action, list_tasks


# ---------------------------------------------------------------------------
# Hardcoded policies — one action generator per task
# ---------------------------------------------------------------------------

def policy_easy_gdv() -> Iterator[dict]:
    """Clear GDV emergency — check vitals, exam abdomen, route, stabilise, admit."""
    yield {"tool": "check_vitals",       "parameters": {"systems": ["cardiovascular", "respiratory"]}}
    yield {"tool": "physical_exam",      "parameters": {"region": "abdomen", "depth": "quick"}}
    yield {"tool": "decide_triage_route","parameters": {"urgency": "immediate_resuscitation",
                                                         "primary_concern": "GDV with haemodynamic compromise"}}
    yield {"tool": "contact_owner",      "parameters": {"purpose": "consent"}}
    yield {"tool": "place_iv_access",    "parameters": {"site": "cephalic"}}
    yield {"tool": "administer_fluid_bolus", "parameters": {"fluid_type": "crystalloid",
                                                             "dose_ml_kg": 10, "rate": "moderate"}}
    yield {"tool": "perform_procedure",  "parameters": {"procedure": "gastric_decompression"}}
    yield {"tool": "make_disposition",   "parameters": {"decision": "admit_icu",
                                                         "aftercare": "Emergency surgery consult"}}


def policy_medium_hcm_cat() -> Iterator[dict]:
    """HCM cat — check vitals, thorax imaging, oxygen, furosemide, admit ward."""
    yield {"tool": "check_vitals",       "parameters": {"systems": ["respiratory", "cardiovascular"]}}
    yield {"tool": "physical_exam",      "parameters": {"region": "thorax", "depth": "quick"}}
    yield {"tool": "physical_exam",      "parameters": {"region": "cardiovascular", "depth": "quick"}}
    yield {"tool": "run_imaging",        "parameters": {"modality": "radiograph", "region": "thorax",
                                                         "priority": "urgent"}}
    yield {"tool": "contact_owner",      "parameters": {"purpose": "consent"}}
    yield {"tool": "decide_triage_route","parameters": {"urgency": "urgent_stabilise",
                                                         "primary_concern": "HCM with pleural effusion"}}
    yield {"tool": "oxygen_therapy",     "parameters": {"method": "oxygen_cage"}}
    yield {"tool": "place_iv_access",    "parameters": {"site": "cephalic"}}
    yield {"tool": "give_medication",    "parameters": {"drug": "furosemide", "dose_mg_kg": 1.0,
                                                         "route": "iv"}}
    # Collect imaging result (ordered at step 3, eta=2, ready by step 5)
    yield {"tool": "check_vitals",       "parameters": {"systems": ["respiratory"]}}
    yield {"tool": "make_disposition",   "parameters": {"decision": "admit_ward",
                                                         "follow_up_hours": 24}}


def policy_hard_polytrauma() -> Iterator[dict]:
    """Polytrauma — urgent imaging both regions, colloid, thoracocentesis, ICU."""
    yield {"tool": "check_vitals",       "parameters": {"systems": ["all"]}}
    yield {"tool": "run_imaging",        "parameters": {"modality": "radiograph", "region": "thorax",
                                                         "priority": "urgent"}}
    yield {"tool": "run_imaging",        "parameters": {"modality": "radiograph", "region": "abdomen",
                                                         "priority": "urgent"}}
    yield {"tool": "contact_owner",      "parameters": {"purpose": "consent"}}
    yield {"tool": "decide_triage_route","parameters": {"urgency": "immediate_resuscitation",
                                                         "primary_concern": "polytrauma — pneumothorax + haemoabdomen"}}
    yield {"tool": "place_iv_access",    "parameters": {"site": "jugular"}}
    yield {"tool": "administer_fluid_bolus", "parameters": {"fluid_type": "colloid",
                                                             "dose_ml_kg": 5, "rate": "moderate"}}
    yield {"tool": "perform_procedure",  "parameters": {"procedure": "thoracocentesis"}}
    yield {"tool": "oxygen_therapy",     "parameters": {"method": "mask"}}
    yield {"tool": "give_medication",    "parameters": {"drug": "methadone", "dose_mg_kg": 0.2,
                                                         "route": "iv"}}
    yield {"tool": "make_disposition",   "parameters": {"decision": "admit_icu",
                                                         "aftercare": "Surgery consult for haemoabdomen"}}


def policy_hard_imha_budget() -> Iterator[dict]:
    """IMHA with tight ₹38,000 budget — CBC+lactate only, skip expensive panels."""
    yield {"tool": "check_vitals",        "parameters": {"systems": ["cardiovascular", "respiratory"]}}
    yield {"tool": "contact_owner",        "parameters": {"purpose": "budget"}}
    yield {"tool": "run_bloodwork",        "parameters": {"panel": "cbc",     "priority": "urgent"}}
    yield {"tool": "run_bloodwork",        "parameters": {"panel": "lactate", "priority": "urgent"}}
    yield {"tool": "decide_triage_route",  "parameters": {"urgency": "urgent_stabilise",
                                                           "primary_concern": "IMHA — icteric, tachycardic"}}
    yield {"tool": "place_iv_access",      "parameters": {"site": "jugular"}}
    yield {"tool": "administer_fluid_bolus","parameters": {"fluid_type": "blood_products",
                                                            "dose_ml_kg": 10, "rate": "slow"}}
    yield {"tool": "give_medication",      "parameters": {"drug": "dexmedetomidine", "dose_mg_kg": 5.0,
                                                           "route": "iv"}}
    yield {"tool": "make_disposition",     "parameters": {"decision": "admit_icu",
                                                           "aftercare": "Monitor PCV, repeat CBC in 4h"}}


def policy_hard_stochastic_pancreatitis_open_loop() -> Iterator[dict]:
    """
    OPEN-LOOP policy — never checks action_succeeded.
    Repeatedly uses oxygen mask without detecting when it's silently removed.
    Scores 0.00 on stochastic_awareness despite 'administering' O2 every step.
    """
    yield {"tool": "check_vitals",        "parameters": {"systems": ["cardiovascular", "respiratory"]}}
    yield {"tool": "physical_exam",        "parameters": {"region": "abdomen", "depth": "quick"}}
    yield {"tool": "contact_owner",        "parameters": {"purpose": "consent"}}
    yield {"tool": "place_iv_access",      "parameters": {"site": "cephalic"}}
    yield {"tool": "oxygen_therapy",       "parameters": {"method": "mask"}}          # SILENT failure risk
    yield {"tool": "decide_triage_route",  "parameters": {"urgency": "urgent_stabilise",
                                                           "primary_concern": "severe pancreatitis"}}
    yield {"tool": "give_medication",      "parameters": {"drug": "morphine",   "dose_mg_kg": 0.3, "route": "im"}}
    yield {"tool": "give_medication",      "parameters": {"drug": "maropitant", "dose_mg_kg": 1.0, "route": "im"}}
    yield {"tool": "oxygen_therapy",       "parameters": {"method": "mask"}}          # SILENT failure risk — ignored
    yield {"tool": "give_medication",      "parameters": {"drug": "morphine",   "dose_mg_kg": 0.3, "route": "im"}}
    yield {"tool": "oxygen_therapy",       "parameters": {"method": "mask"}}          # SILENT failure risk — ignored
    yield {"tool": "give_medication",      "parameters": {"drug": "maropitant", "dose_mg_kg": 1.0, "route": "im"}}
    yield {"tool": "oxygen_therapy",       "parameters": {"method": "mask"}}          # SILENT failure risk — ignored
    yield {"tool": "give_medication",      "parameters": {"drug": "morphine",   "dose_mg_kg": 0.3, "route": "im"}}
    yield {"tool": "oxygen_therapy",       "parameters": {"method": "mask"}}          # SILENT failure risk — ignored
    yield {"tool": "make_disposition",     "parameters": {"decision": "admit_ward", "follow_up_hours": 24}}


def policy_hard_stochastic_pancreatitis_closed_loop() -> Iterator[dict]:
    """
    CLOSED-LOOP policy — reads action_succeeded after each intervention.
    When oxygen mask is silently removed, switches to oxygen_cage next step.
    When IV placement fails, retries at alternate site.
    Scores higher on stochastic_awareness (closed-loop adaptation detected).
    """
    yield {"tool": "check_vitals",        "parameters": {"systems": ["cardiovascular", "respiratory"]}}
    yield {"tool": "physical_exam",        "parameters": {"region": "abdomen", "depth": "quick"}}
    yield {"tool": "contact_owner",        "parameters": {"purpose": "consent"}}
    # Closed-loop IV — retry logic handled in run_stochastic_episode
    yield {"tool": "place_iv_access",      "parameters": {"site": "cephalic"}}
    # Oxygen — will switch to oxygen_cage if mask silently fails (see run_stochastic_episode)
    yield {"tool": "oxygen_therapy",       "parameters": {"method": "mask"}}
    yield {"tool": "decide_triage_route",  "parameters": {"urgency": "urgent_stabilise",
                                                           "primary_concern": "pancreatitis with uncooperative patient"}}
    yield {"tool": "give_medication",      "parameters": {"drug": "morphine",   "dose_mg_kg": 0.3, "route": "iv"}}
    yield {"tool": "give_medication",      "parameters": {"drug": "maropitant", "dose_mg_kg": 1.0, "route": "iv"}}
    yield {"tool": "make_disposition",     "parameters": {"decision": "admit_ward", "follow_up_hours": 24}}


POLICIES = {
    "easy_gdv":                             policy_easy_gdv,
    "medium_hcm_cat":                       policy_medium_hcm_cat,
    "hard_imha_budget":                     policy_hard_imha_budget,
    "hard_polytrauma":                      policy_hard_polytrauma,
    "hard_stochastic_pancreatitis":         policy_hard_stochastic_pancreatitis_closed_loop,
    # Open-loop variant for comparison — shows stochastic_awareness = 0.000
    "hard_stochastic_pancreatitis_open":    policy_hard_stochastic_pancreatitis_open_loop,
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_policy_episode(env: VetTriageEnv, task_id: str, seed: int, verbose: bool,
                       closed_loop: bool = False) -> dict:
    """
    Run one episode with a scripted policy.

    closed_loop: if True, reads obs.action_succeeded after each step and adapts:
      - oxygen mask silently failed → switch to oxygen_cage on next attempt
      - IV access failed → retry at saphenous site
    This demonstrates the stochastic_awareness mechanic.
    """
    obs = env.reset(task_id=task_id, seed=seed)

    if verbose:
        print(f"  Patient: {obs.species} | {obs.breed} | {obs.age_years}yr")
        print(f"  Complaint: {obs.presenting_complaint[:80]}")

    policy_gen = POLICIES[task_id]()
    total_reward = 0.0
    done = False
    info = {}
    steps = 0

    # Closed-loop state
    oxy_method = "mask"
    iv_site = "cephalic"

    for action_dict in policy_gen:
        if done:
            break

        params = dict(action_dict["parameters"])

        # Closed-loop: apply adapted state to oxygen/iv actions
        if closed_loop:
            if action_dict["tool"] == "oxygen_therapy":
                params["method"] = oxy_method
            elif action_dict["tool"] == "place_iv_access":
                params["site"] = iv_site

        action = Action(tool=action_dict["tool"], parameters=params)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

        if verbose:
            status = "" if obs.action_succeeded else " [FAILED]"
            print(f"  [Step {steps}] {action.tool}({str(action.parameters)[:50]})"
                  f"{status} r={reward.value:+.3f}")
            if not obs.action_succeeded and obs.latest_clinical_event:
                print(f"    ! Clinical event: {obs.latest_clinical_event[:75]}")
            if obs.events:
                for evt in obs.events:
                    print(f"    ! {evt[:80]}")

        # Closed-loop adaptation based on action_succeeded
        if closed_loop and not obs.action_succeeded:
            ft = env._action_log[-1].get("fail_type", "")
            if action.tool == "oxygen_therapy" and ft == "silent" and oxy_method == "mask":
                oxy_method = "oxygen_cage"
                if verbose:
                    print(f"    [CLOSED-LOOP] Silent mask failure → switching to oxygen_cage")
            elif action.tool == "place_iv_access":
                iv_site = "saphenous" if iv_site == "cephalic" else "jugular"
                if verbose:
                    print(f"    [CLOSED-LOOP] IV failed → retry at {iv_site}")

    return {
        "task_id": task_id,
        "grade": info.get("grade", 0.0),
        "passed": info.get("passed", False),
        "total_reward": round(total_reward, 3),
        "steps": steps,
        "terminal_reason": info.get("terminal_reason", "unknown"),
        "feedback": info.get("grade_feedback", []),
        "breakdown": info.get("grade_breakdown", {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STOCHASTIC_TASK = "hard_stochastic_pancreatitis"


def main():
    parser = argparse.ArgumentParser(description="VetTriageEnv rule-based baseline (no API key)")
    parser.add_argument("--task", default=None, help="Specific task ID (or all if omitted)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    env = VetTriageEnv(max_total_steps=80)
    # Use registered task IDs only (not the _open variant)
    registered = list_tasks()
    tasks_to_run = [args.task] if args.task else registered
    all_results = []

    print("\nVetTriageEnv Rule-Based Baseline (no API key required)")
    print("=" * 60)

    for task_id in tasks_to_run:
        if task_id not in POLICIES:
            continue

        print(f"\nTask: {task_id}")
        print("-" * 40)

        # For the stochastic task, run open-loop then closed-loop for comparison
        if task_id == STOCHASTIC_TASK:
            print("  [Stochastic mechanic showcase: open-loop vs closed-loop]")

            print("\n  -- OPEN-LOOP (never reads action_succeeded) --")
            open_result = run_policy_episode(
                env, task_id, args.seed, verbose=not args.quiet, closed_loop=False
            )
            open_stoch = open_result["breakdown"].get("stochastic_awareness", 0)
            print(f"  -> Grade: {open_result['grade']:.3f} | "
                  f"Stochastic awareness: {open_stoch:.3f} | Steps: {open_result['steps']}")

            print("\n  -- CLOSED-LOOP (reads action_succeeded, adapts on silent failure) --")
            closed_result = run_policy_episode(
                env, task_id, args.seed, verbose=not args.quiet, closed_loop=True
            )
            closed_stoch = closed_result["breakdown"].get("stochastic_awareness", 0)
            print(f"  -> Grade: {closed_result['grade']:.3f} | "
                  f"Stochastic awareness: {closed_stoch:.3f} | Steps: {closed_result['steps']}")

            print(f"\n  Stochastic awareness delta: "
                  f"{closed_stoch - open_stoch:+.3f} (closed-loop advantage)")
            result = closed_result  # use closed-loop as the official result
        else:
            result = run_policy_episode(env, task_id, args.seed, verbose=not args.quiet)
            print(f"  -> Grade: {result['grade']:.3f} | Passed: {result['passed']} | Steps: {result['steps']}")
            if result["feedback"]:
                print(f"  Feedback: {'; '.join(result['feedback'])}")

        all_results.append(result)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("-" * 40)
    for r in all_results:
        print(f"  {r['task_id']:30s}: {r['grade']:.3f}")
    overall = sum(r["grade"] for r in all_results) / len(all_results)
    print(f"  {'OVERALL':30s}: {overall:.3f}")

    with open("baseline_results.json", "w") as f:
        json.dump({"backend": "rule_based", "results": all_results,
                   "summary": {r["task_id"]: r["grade"] for r in all_results}}, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()

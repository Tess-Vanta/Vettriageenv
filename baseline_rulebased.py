"""
Rule-based baseline for VetTriageEnv — no API key required.

Uses a deterministic clinical policy for each task and produces
reproducible scores on all 3 benchmark tasks.

Usage:
    python baseline_rulebased.py
    python baseline_rulebased.py --task easy_gdv
"""
from __future__ import annotations

import argparse
import json
from typing import Iterator

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


POLICIES = {
    "easy_gdv":       policy_easy_gdv,
    "medium_hcm_cat": policy_medium_hcm_cat,
    "hard_polytrauma": policy_hard_polytrauma,
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_policy_episode(env: VetTriageEnv, task_id: str, seed: int, verbose: bool) -> dict:
    obs = env.reset(task_id=task_id, seed=seed)

    if verbose:
        print(f"  Patient: {obs.species} | {obs.breed} | {obs.age_years}yr")
        print(f"  Complaint: {obs.presenting_complaint[:80]}")

    policy_gen = POLICIES[task_id]()
    total_reward = 0.0
    done = False
    info = {}
    steps = 0

    for action_dict in policy_gen:
        if done:
            break

        action = Action(tool=action_dict["tool"], parameters=action_dict["parameters"])
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

        if verbose:
            print(f"  [Step {steps}] {action.tool}({str(action.parameters)[:55]})"
                  f" r={reward.value:+.3f}")
            if obs.events:
                for evt in obs.events:
                    print(f"    ! {evt[:80]}")

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

def main():
    parser = argparse.ArgumentParser(description="VetTriageEnv rule-based baseline (no API key)")
    parser.add_argument("--task", default=None, help="Specific task ID (or all if omitted)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    env = VetTriageEnv(max_total_steps=80)
    tasks_to_run = [args.task] if args.task else list_tasks()
    all_results = []

    print("\nVetTriageEnv Rule-Based Baseline (no API key required)")
    print("=" * 60)

    for task_id in tasks_to_run:
        print(f"\nTask: {task_id}")
        print("-" * 40)
        result = run_policy_episode(env, task_id, args.seed, verbose=not args.quiet)
        all_results.append(result)
        print(f"  -> Grade: {result['grade']:.3f} | Passed: {result['passed']} | Steps: {result['steps']}")
        if result["feedback"]:
            print(f"  Feedback: {'; '.join(result['feedback'])}")

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

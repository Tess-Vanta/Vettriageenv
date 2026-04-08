"""
Task graders for VetTriageEnv — scores agent performance 0.0–1.0.
Each grader has deterministic success/failure criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import FullState
from .physiology import DIAGNOSIS_PROFILES


@dataclass
class GradeResult:
    score: float                        # 0.0–1.0
    passed: bool
    breakdown: Dict[str, float]         # component scores
    feedback: List[str]                 # human-readable notes


# ---------------------------------------------------------------------------
# Routing correctness
# ---------------------------------------------------------------------------

CORRECT_ROUTES = {
    "gastric_dilatation_volvulus":   "immediate_resuscitation",
    "hypertrophic_cardiomyopathy":   "urgent_stabilise",
    "feline_urethral_obstruction":   "urgent_stabilise",
    "polytrauma_hbc":                "immediate_resuscitation",
    "pancreatitis_severe":           "urgent_stabilise",
    "immune_haemolytic_anaemia":     "urgent_stabilise",
    "congestive_heart_failure":      "urgent_stabilise",
    "diabetic_ketoacidosis":         "urgent_stabilise",
}

ACCEPTABLE_ROUTES = {
    "gastric_dilatation_volvulus":   ["immediate_resuscitation"],
    "hypertrophic_cardiomyopathy":   ["urgent_stabilise", "immediate_resuscitation"],
    "feline_urethral_obstruction":   ["urgent_stabilise", "immediate_resuscitation"],
    "polytrauma_hbc":                ["immediate_resuscitation"],
    "pancreatitis_severe":           ["urgent_stabilise", "diagnostic_workup"],
    "immune_haemolytic_anaemia":     ["urgent_stabilise", "immediate_resuscitation"],
    "congestive_heart_failure":      ["urgent_stabilise", "immediate_resuscitation"],
    "diabetic_ketoacidosis":         ["urgent_stabilise", "diagnostic_workup"],
}

DANGEROUS_ROUTES = {
    # These earn a large penalty regardless of other performance
    "gastric_dilatation_volvulus":   ["discharge", "owner_decline_treatment"],
    "polytrauma_hbc":                ["discharge", "owner_decline_treatment"],
    "feline_urethral_obstruction":   ["discharge", "diagnostic_workup"],
    "hypertrophic_cardiomyopathy":   ["discharge"],
}

CORRECT_DISPOSITIONS = {
    "gastric_dilatation_volvulus":   ["admit_icu", "refer_specialist"],
    "hypertrophic_cardiomyopathy":   ["admit_icu", "admit_ward"],
    "feline_urethral_obstruction":   ["admit_icu", "admit_ward"],
    "polytrauma_hbc":                ["admit_icu", "refer_specialist"],
    "pancreatitis_severe":           ["admit_ward", "admit_icu"],
    "immune_haemolytic_anaemia":     ["admit_icu", "admit_ward"],
    "congestive_heart_failure":      ["admit_ward", "admit_icu"],
    "diabetic_ketoacidosis":         ["admit_ward", "admit_icu"],
}

KEY_EXAMINATIONS = {
    "gastric_dilatation_volvulus": [
        ("check_vitals", "cardiovascular"),
        ("physical_exam", "abdomen"),
    ],
    "hypertrophic_cardiomyopathy": [
        ("check_vitals", "respiratory"),
        ("physical_exam", "thorax"),
        ("run_imaging", "radiograph_thorax"),
    ],
    "feline_urethral_obstruction": [
        ("check_vitals", "cardiovascular"),
        ("physical_exam", "abdomen"),
        ("run_bloodwork", "chemistry"),  # potassium
    ],
    "polytrauma_hbc": [
        ("check_vitals", "cardiovascular"),
        ("run_imaging", "radiograph_thorax"),
        ("run_imaging", "radiograph_abdomen"),
    ],
    "pancreatitis_severe": [
        ("physical_exam", "abdomen"),
        ("run_bloodwork", "chemistry"),
    ],
    "immune_haemolytic_anaemia": [
        ("check_vitals", "cardiovascular"),
        ("run_bloodwork", "cbc"),
    ],
    "congestive_heart_failure": [
        ("check_vitals", "respiratory"),
        ("physical_exam", "cardiovascular"),
        ("run_imaging", "radiograph_thorax"),
    ],
    "diabetic_ketoacidosis": [
        ("run_bloodwork", "chemistry"),
        ("run_bloodwork", "blood_gas"),
    ],
}

HARMFUL_ACTIONS = {
    "gastric_dilatation_volvulus":   ["discharge", "make_disposition:discharge_with_medication",
                                       "make_disposition:discharge_no_treatment"],
    "polytrauma_hbc":                ["discharge", "administer_fluid_bolus:crystalloid"],
    "hypertrophic_cardiomyopathy":   ["administer_fluid_bolus:crystalloid"],
    "feline_urethral_obstruction":   ["administer_fluid_bolus", "discharge"],
}


def grade_episode(state: FullState, meta: dict, action_log: list) -> GradeResult:
    """
    Score a completed episode.
    Returns GradeResult with score 0.0–1.0.
    """
    diag = state.patient.true_diagnosis
    profile = DIAGNOSIS_PROFILES.get(diag)
    breakdown = {}
    feedback = []
    total_steps = state.step

    # --- 1. Triage route correctness (0–0.30) ---
    route = state.triage_route_decided
    correct_route = CORRECT_ROUTES.get(diag, "urgent_stabilise")
    acceptable = ACCEPTABLE_ROUTES.get(diag, [correct_route])
    dangerous = DANGEROUS_ROUTES.get(diag, [])

    if route == correct_route:
        route_score = 0.30
        feedback.append(f"Correct triage route: {route}")
    elif route in acceptable:
        route_score = 0.18
        feedback.append(f"Acceptable triage route: {route} (optimal: {correct_route})")
    elif route in dangerous:
        route_score = 0.0
        feedback.append(f"DANGEROUS route: {route} for {diag} — patient at risk")
    elif route is None:
        route_score = 0.0
        feedback.append("No triage route decided")
    else:
        route_score = 0.08
        feedback.append(f"Suboptimal route: {route}")
    breakdown["route_correctness"] = route_score

    # --- 2. Disposition correctness (0–0.25) ---
    disposition = state.disposition_made
    correct_disps = CORRECT_DISPOSITIONS.get(diag, ["admit_ward"])

    if disposition in correct_disps:
        disp_score = 0.25
        feedback.append(f"Correct disposition: {disposition}")
    elif disposition in ("discharge_no_treatment", "discharge_with_medication") and route in dangerous:
        disp_score = 0.0
        feedback.append(f"CRITICAL ERROR: {diag} patient discharged")
    elif disposition is None:
        disp_score = 0.0
        feedback.append("No disposition made")
    else:
        disp_score = 0.08
        feedback.append(f"Suboptimal disposition: {disposition} (expected: {correct_disps[0]})")
    breakdown["disposition_correctness"] = disp_score

    # --- 3. Key examinations performed (0–0.20) ---
    key_exams = KEY_EXAMINATIONS.get(diag, [])
    exams_done = _extract_exams_done(action_log)
    exams_hit = 0
    for exam_tool, exam_target in key_exams:
        if _exam_was_done(exams_done, exam_tool, exam_target):
            exams_hit += 1
    exam_score = (exams_hit / max(len(key_exams), 1)) * 0.20
    breakdown["key_examinations"] = exam_score
    feedback.append(f"Key exams: {exams_hit}/{len(key_exams)} performed")

    # --- 4. Efficiency (0–0.15) — fewer steps is better ---
    # Maximum useful steps: 30 for easy, 40 for medium, 50 for hard
    difficulty = meta.get("difficulty", "medium")
    step_limits = {"easy": 15, "medium": 25, "hard": 40}
    step_budget = step_limits.get(difficulty, 25)
    if total_steps <= step_budget * 0.6:
        eff_score = 0.15
        feedback.append(f"Excellent efficiency: {total_steps} steps")
    elif total_steps <= step_budget:
        eff_score = 0.10
        feedback.append(f"Good efficiency: {total_steps} steps")
    elif total_steps <= step_budget * 1.5:
        eff_score = 0.05
        feedback.append(f"Over-examined: {total_steps} steps")
    else:
        eff_score = 0.0
        feedback.append(f"Severely over-examined: {total_steps} steps (budget: {step_budget})")
    breakdown["efficiency"] = eff_score

    # --- 4b. Time-awareness (0–0.10) — reward early intervention ---
    # Find the sim_time_hours when the first therapeutic action was taken.
    # Therapeutic = IV, fluid, medication, oxygen, procedure (not exams or consults).
    first_treat_time = _find_first_treatment_time(action_log, state)
    sim_total = state.sim_time_hours
    if first_treat_time is not None:
        if first_treat_time <= 0.30:
            time_score = 0.10
            feedback.append(f"Rapid first treatment at T+{first_treat_time:.2f}h — excellent time-awareness")
        elif first_treat_time <= 0.60:
            time_score = 0.06
            feedback.append(f"Adequate first treatment at T+{first_treat_time:.2f}h")
        elif first_treat_time <= 1.0:
            time_score = 0.02
            feedback.append(f"Delayed first treatment at T+{first_treat_time:.2f}h — patient deteriorated while waiting")
        else:
            time_score = 0.0
            feedback.append(
                f"Critically late first treatment at T+{first_treat_time:.2f}h — "
                f"waited too long for diagnostics"
            )
    else:
        time_score = 0.0
        feedback.append("No therapeutic intervention performed")
    breakdown["time_awareness"] = time_score

    # --- 5. Patient outcome (0–0.10) ---
    final_severity = state.patient.severity
    if final_severity < 0.45:
        outcome_score = 0.10
        feedback.append("Patient stable at end of episode")
    elif final_severity < 0.65:
        outcome_score = 0.06
        feedback.append("Patient moderately compromised")
    elif final_severity < profile.lethal_threshold if profile else 0.90:
        outcome_score = 0.02
        feedback.append("Patient severely compromised — delayed intervention")
    else:
        outcome_score = 0.0
        feedback.append("Patient deteriorated to critical level")
    breakdown["patient_outcome"] = outcome_score

    # --- Dangerous action penalty ---
    harmful_found = _check_harmful_actions(action_log, diag)
    if harmful_found:
        penalty = -0.10 * len(harmful_found)
        breakdown["harmful_action_penalty"] = penalty
        feedback.extend([f"Harmful action: {a}" for a in harmful_found])
    else:
        breakdown["harmful_action_penalty"] = 0.0

    # --- Budget efficiency (0–0.10) ---
    # Reward agents that complete the task under budget AND avoid wasteful testing.
    budget_limit = state.owner.budget_limit
    budget_spent = state.owner.budget_spent
    if budget_limit is None:
        # No budget constraint — small neutral score
        breakdown["budget_efficiency"] = 0.03
        feedback.append("No budget constraint — budget efficiency not scored")
    elif budget_spent > budget_limit:
        overspend = budget_spent - budget_limit
        breakdown["budget_efficiency"] = 0.0
        feedback.append(f"OVER BUDGET: spent ₹{budget_spent:.0f} of ₹{budget_limit:.0f} limit (overspend ₹{overspend:.0f})")
    else:
        utilisation = budget_spent / budget_limit
        if utilisation <= 0.60:
            budget_score = 0.10
            feedback.append(f"Excellent budget efficiency: spent ₹{budget_spent:.0f} of ₹{budget_limit:.0f} ({utilisation:.0%})")
        elif utilisation <= 0.85:
            budget_score = 0.06
            feedback.append(f"Good budget management: spent ₹{budget_spent:.0f} of ₹{budget_limit:.0f} ({utilisation:.0%})")
        elif utilisation <= 1.0:
            budget_score = 0.03
            feedback.append(f"Near budget limit: spent ₹{budget_spent:.0f} of ₹{budget_limit:.0f} ({utilisation:.0%})")
        else:
            budget_score = 0.0
        breakdown["budget_efficiency"] = budget_score

    total = sum(breakdown.values())
    total = max(0.0, min(1.0, total))

    return GradeResult(
        score=round(total, 3),
        passed=total >= 0.55,
        breakdown=breakdown,
        feedback=feedback,
    )


def _extract_exams_done(action_log: list) -> list:
    """Extract (tool, key_param) from action log."""
    done = []
    for entry in action_log:
        if isinstance(entry, dict):
            tool = entry.get("tool", "")
            params = entry.get("params", {})
            done.append((tool, params))
        elif isinstance(entry, str):
            done.append((entry, {}))
    return done


def _exam_was_done(exams_done: list, tool: str, target: str) -> bool:
    """Check if a specific exam was performed."""
    for t, params in exams_done:
        if t == tool:
            # Check if target appears in params values
            for v in params.values():
                if isinstance(v, str) and target in v:
                    return True
                if isinstance(v, list) and any(target in str(x) for x in v):
                    return True
            # Also match panel-level
            if tool == "run_bloodwork" and params.get("panel") == target:
                return True
            if tool == "run_bloodwork" and params.get("panel") == "full_panel":
                return True
            if tool == "run_imaging":
                combo = f"{params.get('modality','')}_{ params.get('region','')}"
                if target == combo or target in combo:
                    return True
            if tool == "check_vitals":
                systems = params.get("systems", [])
                if target in systems or "all" in systems:
                    return True
            if tool == "physical_exam":
                if params.get("region") == target:
                    return True
    return False


def _check_harmful_actions(action_log: list, diag: str) -> list:
    """Return list of harmful actions that were taken."""
    harmful_list = HARMFUL_ACTIONS.get(diag, [])
    found = []
    for entry in action_log:
        if isinstance(entry, dict):
            tool = entry.get("tool", "")
            params = entry.get("params", {})
            for h in harmful_list:
                if ":" in h:
                    h_tool, h_param = h.split(":", 1)
                    if tool == h_tool and h_param in str(params):
                        found.append(h)
                elif tool == h:
                    found.append(h)
    return found


# ---------------------------------------------------------------------------
# Per-step reward function
# ---------------------------------------------------------------------------

def compute_step_reward(
    action: dict,
    state_before: FullState,
    state_after: FullState,
    obs_updates: dict,
    meta: dict,
    action_log: list,
) -> tuple:
    """
    Returns (reward_value, components_dict, message).
    Called every step to provide dense reward signal.
    """
    diag = state_before.patient.true_diagnosis
    components = {}
    messages = []

    tool = action.get("tool", "")
    params = action.get("params", {})

    # --- Informative examination reward ---
    if tool in ("check_vitals", "physical_exam", "run_bloodwork", "run_imaging"):
        # Reward if this is the first time this exam/region was done
        is_repeat = _is_repeat_action(tool, params, action_log[:-1])  # exclude current
        if not is_repeat:
            # Check if this exam is a key examination for this diagnosis
            key_exams = KEY_EXAMINATIONS.get(diag, [])
            if _exam_was_done([(tool, params)], tool, _get_exam_target(tool, params)):
                for kt, kp in key_exams:
                    if kt == tool and kp in str(params):
                        components["informative_exam"] = +0.3
                        messages.append(f"Informative exam: {tool}")
                        break
                else:
                    components["informative_exam"] = +0.05
            else:
                components["informative_exam"] = +0.05
        else:
            components["redundant_exam"] = -0.15
            messages.append(f"Redundant exam: {tool}")

    # --- IV placement when needed ---
    if tool == "place_iv_access" and not state_before.patient.iv_access:
        if state_before.patient.severity > 0.4:
            components["iv_access_when_needed"] = +0.2
            messages.append("IV access placed appropriately")

    # --- Oxygen when hypoxic ---
    if tool == "oxygen_therapy" and state_before.patient.spo2 < 94:
        components["appropriate_oxygen"] = +0.15
        messages.append("Oxygen provided for hypoxaemia")

    # --- Time pressure penalty when critical ---
    severity = state_before.patient.severity
    if severity >= 0.65:
        components["time_penalty"] = -0.15
        messages.append("Patient critical — time penalty")
    elif severity >= 0.50:
        components["time_penalty"] = -0.05

    # --- Time-delay penalty: non-therapeutic actions while patient is deteriorating ---
    # Penalise "waiting" actions (consults, owner calls) when SpO2 < 90 and no treatment yet
    spo2 = state_before.patient.spo2
    if spo2 < 90 and tool in ("contact_owner", "consult_specialist", "run_bloodwork", "run_imaging"):
        # Only penalise if no therapeutic action has been taken yet
        has_treated = any(
            isinstance(e, dict) and e.get("tool", "") in THERAPEUTIC_TOOLS
            for e in action_log[:-1]
        )
        if not has_treated:
            components["waiting_while_hypoxic"] = -0.20
            messages.append(
                f"SpO2 {spo2:.0f}% — ordering tests/consulting instead of treating"
            )

    # --- Budget overspend penalty ---
    if state_before.owner.budget_limit:
        spent = state_after.owner.budget_spent
        limit = state_before.owner.budget_limit
        if spent > limit:
            overage = spent - limit
            components["budget_penalty"] = -min(0.4, overage / 30000)
            messages.append(f"Over budget by ₹{overage:.0f}")
        elif limit - spent < 5000 and limit < 60000:
            # Near budget on a tight-budget task — warn
            components["near_budget_warning"] = -0.02
            messages.append(f"Budget nearly exhausted: ₹{spent:.0f}/₹{limit:.0f}")

    # --- Dangerous action penalty ---
    harmful = HARMFUL_ACTIONS.get(diag, [])
    for h in harmful:
        h_tool = h.split(":")[0] if ":" in h else h
        if tool == h_tool:
            components["harmful_action"] = -0.30
            messages.append(f"Harmful action for {diag}: {tool}")
            break

    total = sum(components.values())
    msg = "; ".join(messages) if messages else "Step completed"
    return round(total, 3), components, msg


THERAPEUTIC_TOOLS = {
    "place_iv_access", "administer_fluid_bolus", "give_medication",
    "oxygen_therapy", "perform_procedure",
}


def _find_first_treatment_time(action_log: list, state) -> Optional[float]:
    """Find sim_time_hours when the first therapeutic action was taken.

    Walks the action_history strings from FullState to find the first
    therapeutic tool, then estimates sim_time from the step index.
    Falls back to scanning action_log dicts.
    """
    from .tools import ACTION_TIME_HOURS
    # Scan action_log (list of dicts with 'tool' key) and accumulate time
    cumulative_time = 0.0
    for entry in action_log:
        if isinstance(entry, dict):
            tool = entry.get("tool", "")
            cumulative_time += ACTION_TIME_HOURS.get(tool, 0.083)
            if tool in THERAPEUTIC_TOOLS:
                return round(cumulative_time, 3)
    return None


def _is_repeat_action(tool: str, params: dict, prior_log: list) -> bool:
    """Check if the same tool+params appeared in prior log."""
    for entry in prior_log[-6:]:  # only look at recent history
        if isinstance(entry, dict):
            if entry.get("tool") == tool and entry.get("params") == params:
                return True
    return False


def _get_exam_target(tool: str, params: dict) -> str:
    if tool == "check_vitals":
        systems = params.get("systems", [])
        return systems[0] if systems else "all"
    if tool == "physical_exam":
        return params.get("region", "")
    if tool == "run_bloodwork":
        return params.get("panel", "")
    if tool == "run_imaging":
        return f"{params.get('modality','')}_{params.get('region','')}"
    return ""

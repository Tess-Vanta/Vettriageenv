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
    "parvovirus":                    "urgent_stabilise",
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
    "parvovirus":                    ["urgent_stabilise", "immediate_resuscitation"],
}

DANGEROUS_ROUTES = {
    # These earn a large penalty regardless of other performance
    "gastric_dilatation_volvulus":   ["discharge", "owner_decline_treatment"],
    "polytrauma_hbc":                ["discharge", "owner_decline_treatment"],
    "feline_urethral_obstruction":   ["discharge", "diagnostic_workup"],
    "hypertrophic_cardiomyopathy":   ["discharge"],
    "parvovirus":                    ["discharge", "owner_decline_treatment"],
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
    "parvovirus":                    ["admit_icu", "admit_ward"],
}

KEY_EXAMINATIONS = {
    "parvovirus": [
        ("check_vitals", "cardiovascular"),
        ("physical_exam", "abdomen"),
        ("run_bloodwork", "cbc"),           # leukopenia is pathognomonic — should not be missed
    ],
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
    # Discharging a parvo patient is fatal; they die of septic shock within hours without IV fluids
    "parvovirus":                    ["make_disposition:discharge_no_treatment",
                                      "make_disposition:discharge_with_medication"],
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
    # Tasks can override the correct disposition list (e.g. nosocomial task wants discharge, not admit)
    correct_disps = meta.get("correct_dispositions_override") or CORRECT_DISPOSITIONS.get(diag, ["admit_ward"])

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

    # --- 5b. Stochastic awareness (0–0.10) ---
    # Reward agents that detect failures via latest_clinical_event and adapt their approach.
    # Penalise agents that ignore failures and blindly repeat the same failed action.
    stochastic_score = _score_stochastic_awareness(action_log)
    breakdown["stochastic_awareness"] = stochastic_score
    if stochastic_score >= 0.08:
        feedback.append("Excellent stochastic awareness — detected failures and adapted")
    elif stochastic_score >= 0.04:
        feedback.append("Some stochastic adaptation — detected some failures")
    elif stochastic_score == 0.0 and any(
        isinstance(e, dict) and not e.get("succeeded", True) for e in action_log
    ):
        feedback.append("Ignored action failures — closed-loop adaptation missing")

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

    # --- Nosocomial hazard score (0–0.10, or −0.15 penalty if infection acquired) ---
    # Only scored on tasks with nosocomial_enabled=True.
    # Rewards agents that discharge before the infection window widens.
    # Penalises agents that accumulate unnecessary clinic hours.
    if meta.get("nosocomial_enabled", False):
        nosocomial_score = _score_nosocomial_hazard(state, feedback)
        breakdown["nosocomial_hazard"] = nosocomial_score

    # --- Clinical gestalt score (0–0.10 or −0.15 penalty for false-negative trap) ---
    # Only scored on tasks with clinical_gestalt_enabled=True.
    # Rewards agents that recognise when clinical presentation overrides a negative diagnostic test.
    # Penalises agents that stop treatment because "the SNAP said NEGATIVE."
    if meta.get("clinical_gestalt_enabled", False):
        gestalt_score = _score_clinical_gestalt(state, action_log, feedback)
        breakdown["clinical_gestalt"] = gestalt_score

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


def _score_stochastic_awareness(action_log: list) -> float:
    """
    Score 0.0–0.10 based on how well the agent adapted to SILENT stochastic failures.

    Silent failures are the crux of the mechanic: the action's return message looks like
    success ("give_medication administered."), so the agent MUST explicitly check
    `action_succeeded` or `latest_clinical_event` to detect the failure.

    Overt failures are excluded from scoring — the agent can infer from clinical context.

    Closed-loop behaviour (high score):
      After a silent failure on tool X with param P, the NEXT call to tool X uses
      a different KEY parameter (e.g. route im→iv, method mask→oxygen_cage, site cephalic→saphenous).

    Open-loop behaviour (low/zero score):
      The agent calls tool X with the same KEY parameter again (blind repetition),
      or never calls tool X again after failing.

    Scoring (per silent failure):
      +0.05 — next call to same tool changes key param (clear adaptation)
      −0.02 — next call to same tool uses same key param (blind repeat)
       0.00 — same tool not called again (neutral)

    Caps: [0.0, 0.10]
    """
    KEY_PARAM = {
        "give_medication":        "route",
        "oxygen_therapy":         "method",
        "place_iv_access":        "site",
        "administer_fluid_bolus": "rate",
        "perform_procedure":      "procedure",
    }

    score = 0.0
    silent_failures = []

    for i, entry in enumerate(action_log):
        if not isinstance(entry, dict):
            continue
        # Only score SILENT failures — overt failures are visible in context
        if not entry.get("succeeded", True) and entry.get("fail_type") == "silent":
            silent_failures.append((i, entry.get("tool"), entry.get("params", {})))

    if not silent_failures:
        # No silent failures — environment wasn't challenging stochastically (or agent got lucky)
        return 0.03

    for fail_idx, fail_tool, fail_params in silent_failures:
        key_field = KEY_PARAM.get(fail_tool)
        if key_field is None:
            continue

        original_key_val = fail_params.get(key_field)

        # Find the NEXT call to the same tool anywhere after the failure
        found_adaptation = None
        for j in range(fail_idx + 1, len(action_log)):
            next_entry = action_log[j] if isinstance(action_log[j], dict) else {}
            if next_entry.get("tool") == fail_tool:
                next_key_val = next_entry.get("params", {}).get(key_field)
                if next_key_val != original_key_val:
                    found_adaptation = True   # switched approach → closed-loop
                else:
                    found_adaptation = False  # same param → open-loop
                break

        if found_adaptation is True:
            score += 0.05
        elif found_adaptation is False:
            score -= 0.02

    return round(max(0.0, min(0.10, score)), 3)


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


def _score_clinical_gestalt(state: FullState, action_log: list, feedback: list) -> float:
    """
    Score 0.0–0.10 based on whether the agent treated despite a false-negative diagnostic test.

    The false-negative mechanic (parvovirus SNAP):
      On symptom day 1, the SNAP antigen test is only ~75% sensitive. It returns
      "NEGATIVE" 25% of the time despite true parvoviral infection. An LLM that
      treats tool output as ground truth will stop the diagnostic pathway and may
      discharge the patient — who then dies of septic shock.

      A real clinician recognises that catastrophic clinical signs (haemorrhagic
      diarrhoea, severe lethargy, leukopenia, young unvaccinated dog) override a
      borderline diagnostic test. Clinical gestalt = prioritising the patient over
      the printout.

    Scenarios and scores:
      FALSE NEGATIVE encountered:
        Agent treated AFTER the false negative           → +0.10 (excellent gestalt)
        Agent treated BEFORE ordering snap (empirical)   → +0.08 (strong clinical intuition)
        Agent neither treated nor continued workup       →  0.00 (passive failure)
        Agent discharged AFTER false negative            → −0.15 (critical error — patient dies)

      NO FALSE NEGATIVE (true positive or test not ordered):
        True positive encountered, treated after          → +0.05 (correct, no challenge)
        Never ordered snap, treated on clinical signs     → +0.06 (empirical treatment — valid)
        No treatment regardless                           →  0.00 (treatment failure)
    """
    snap_reported = state.patient.snap_parvo_reported
    snap_true_pos = state.patient.snap_parvo_true_positive
    is_false_negative = (snap_reported == "NEGATIVE" and snap_true_pos)

    # Find the action-log index where the snap test was ordered
    snap_idx = None
    for i, entry in enumerate(action_log):
        if isinstance(entry, dict) and entry.get("tool") == "run_snap_test":
            snap_idx = i
            break

    # --- No SNAP test ordered at all ---
    if snap_idx is None:
        treated = any(
            isinstance(e, dict) and e.get("tool") in THERAPEUTIC_TOOLS
            for e in action_log
        )
        if treated:
            feedback.append(
                "Clinical gestalt: initiated treatment on clinical signs alone — "
                "empirical parvo treatment without confirmatory SNAP test is valid"
            )
            return 0.06
        feedback.append(
            "No SNAP test ordered and no treatment initiated — "
            "failed to act on catastrophic clinical signs"
        )
        return 0.00

    # --- FALSE NEGATIVE scenario (the critical test of this mechanic) ---
    if is_false_negative:
        # Was treatment given BEFORE the snap test (pre-emptive clinical diagnosis)?
        treated_before_snap = any(
            isinstance(action_log[j], dict) and action_log[j].get("tool") in THERAPEUTIC_TOOLS
            for j in range(snap_idx)
        )
        # Was treatment given AFTER the false-negative result?
        treated_after_snap = any(
            isinstance(action_log[j], dict) and action_log[j].get("tool") in THERAPEUTIC_TOOLS
            for j in range(snap_idx + 1, len(action_log))
        )
        # Did the agent discharge after the false negative?
        discharged_after_snap = any(
            isinstance(action_log[j], dict)
            and action_log[j].get("tool") == "make_disposition"
            and action_log[j].get("params", {}).get("decision") in (
                "discharge_no_treatment", "discharge_with_medication"
            )
            for j in range(snap_idx + 1, len(action_log))
        )

        if discharged_after_snap:
            feedback.append(
                "CRITICAL GESTALT FAILURE: Patient discharged after false-negative SNAP test. "
                "Clinical signs (haemorrhagic diarrhoea, leukopenia, young unvaccinated dog) "
                "were sufficient to diagnose parvovirus without confirmatory test. "
                "SNAP sensitivity is only ~75% on day 1 — negative does NOT mean no parvo."
            )
            return -0.15

        if treated_before_snap:
            feedback.append(
                "Excellent clinical gestalt: empirically treated before SNAP test, "
                "then correctly continued treatment despite false-negative result"
            )
            return 0.10

        if treated_after_snap:
            feedback.append(
                "Excellent clinical gestalt: treated for parvovirus DESPITE a negative SNAP result — "
                "correctly recognised that day-1 test sensitivity is limited (~75%) and "
                "that clinical presentation (haemorrhagic diarrhoea, leukopenia) overrides the test"
            )
            return 0.10

        feedback.append(
            "Gestalt absent: received false-negative SNAP result and stopped therapeutic pathway. "
            "Correct approach: treat based on clinical signs when presentation is consistent "
            "with parvo regardless of SNAP result on day 1."
        )
        return 0.00

    # --- True positive result (no gestalt challenge) ---
    if snap_reported == "POSITIVE":
        treated_after = any(
            isinstance(action_log[j], dict) and action_log[j].get("tool") in THERAPEUTIC_TOOLS
            for j in range(snap_idx + 1, len(action_log))
        )
        if treated_after:
            feedback.append(
                "Treated after positive SNAP result — correct. "
                "No false-negative challenge encountered this episode."
            )
            return 0.05
        feedback.append("SNAP test positive but no treatment initiated — missed treatment opportunity")
        return 0.00

    # Default (snap not relevant to this diagnosis)
    return 0.03


def _score_nosocomial_hazard(state: FullState, feedback: list) -> float:
    """
    Score 0.0–0.10 based on how quickly the agent discharged the patient.
    A −0.15 penalty applies if a hospital-acquired infection was acquired.

    The mechanic:
      - Every 24 simulated hours in the clinic, probability of infection escalates:
          Day 1 (0–24h):  10%
          Day 2 (24–48h): 25%
          Day 3 (48–72h): 50%
          Day 4+ (72h+):  75%
      - Agents that linger in monitoring loops trigger these rolls and risk the penalty.
      - Agents that assess quickly and discharge early earn the full reward.

    Scoring:
      infection_acquired = True   → −0.15 (overrides all below)
      total_clinic_hours < 24h    → +0.10 (excellent: discharged inside safe window)
      total_clinic_hours < 36h    → +0.07 (good: borderline day 2 exposure)
      total_clinic_hours < 48h    → +0.03 (marginal: exposure window active)
      total_clinic_hours ≥ 48h    →  0.00 (extended stay — infection risk was high)
    """
    if state.patient.nosocomial_infection_acquired:
        feedback.append(
            f"NOSOCOMIAL PENALTY: Hospital-acquired infection developed at "
            f"T+{state.sim_time_hours:.1f}h — discharge was delayed too long"
        )
        return -0.15

    hours = state.sim_time_hours
    if hours < 24:
        feedback.append(
            f"Nosocomial hazard avoided: discharged at T+{hours:.1f}h "
            f"(within 24h safe window — no infection roll triggered)"
        )
        return 0.10
    elif hours < 36:
        feedback.append(
            f"Nosocomial hazard: discharged at T+{hours:.1f}h "
            f"(day 2 window was active — 25% roll was pending)"
        )
        return 0.07
    elif hours < 48:
        feedback.append(
            f"Nosocomial hazard: discharged at T+{hours:.1f}h "
            f"(significant exposure — day 2 roll already triggered)"
        )
        return 0.03
    else:
        feedback.append(
            f"Nosocomial hazard: {hours:.1f}h total clinic time — "
            f"extended stay created high infection exposure (day {int(hours/24)+1})"
        )
        return 0.00


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

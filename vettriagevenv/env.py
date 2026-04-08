"""
VetTriageEnv — Main environment class implementing the OpenEnv interface.

API:
    reset(task_id, seed) -> Observation
    step(action)         -> (Observation, Reward, done, info)
    state()              -> dict (full internal state, for logging/debugging)
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional, Tuple

from .models import (
    Action, Observation, Reward, FullState, VitalsSnapshot, AsyncJob
)
from .physiology import DIAGNOSIS_PROFILES, update_physiology
from .generator import generate_case
from .tools import ToolExecutor, get_available_tools, TOOL_DEFINITIONS, ACTION_TIME_HOURS
from .graders import grade_episode, compute_step_reward
from .tasks import TASK_REGISTRY, get_task


PHASE_ORDER = ["triage", "stabilisation", "monitoring", "disposition"]
DEFAULT_PHASE_LIMITS = {
    "triage": 20,
    "stabilisation": 25,
    "monitoring": 20,
    "disposition": 5,
}


class VetTriageEnv:
    """
    Veterinary triage RL environment.

    An AI agent must gather clinical information via tool calls and make a
    correct triage decision for a procedurally-generated animal patient.

    Implements the OpenEnv interface:
        reset(task_id, seed) -> Observation
        step(action)         -> (Observation, Reward, bool, dict)
        state()              -> dict
    """

    metadata = {
        "name": "VetTriageEnv",
        "version": "1.0.0",
        "tasks": ["easy_gdv", "medium_hcm_cat", "hard_polytrauma"],
        "reward_range": (-5.0, 5.0),
    }

    def __init__(self, max_total_steps: int = 100):
        self.max_total_steps = max_total_steps
        self._state: Optional[FullState] = None
        self._meta: Optional[dict] = None
        self._rng: Optional[random.Random] = None
        self._action_log: list = []
        self._tool_executor = ToolExecutor()
        self._cumulative_reward: float = 0.0
        self._phase_limits: dict = dict(DEFAULT_PHASE_LIMITS)
        self._treatment_effects_pending: list = []

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "random",
        seed: Optional[int] = None,
    ) -> Observation:
        """
        Reset the environment to a new episode.

        Args:
            task_id: one of "easy_gdv", "medium_hcm_cat", "hard_polytrauma",
                     or "random" for procedural generation.
            seed:    random seed for reproducibility.

        Returns:
            Initial Observation.
        """
        self._rng = random.Random(seed)
        self._action_log = []
        self._cumulative_reward = 0.0
        self._treatment_effects_pending = []

        if task_id in TASK_REGISTRY:
            spec = TASK_REGISTRY[task_id]
            self._state, self._meta = generate_case(
                seed=spec.seed if seed is None else seed,
                difficulty=spec.difficulty,
                force_diagnosis=spec.force_diagnosis,
                force_species=spec.force_species,
                task_id=task_id,
            )
            self._phase_limits = dict(DEFAULT_PHASE_LIMITS)
            self._meta["task_spec"] = spec
        else:
            # Procedural random case
            difficulty = "random" if task_id == "random" else task_id
            self._state, self._meta = generate_case(
                seed=seed,
                difficulty=difficulty,
                task_id=task_id,
            )
            self._phase_limits = dict(DEFAULT_PHASE_LIMITS)

        return self._build_observation(events=[])

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(
        self, action: Action
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Take one action in the environment.

        Args:
            action: Action(tool, parameters, reasoning)

        Returns:
            (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        state = self._state
        meta = self._meta
        profile = DIAGNOSIS_PROFILES.get(state.patient.true_diagnosis)

        # --- Validate action ---
        available = get_available_tools(state.phase, state.patient, state.owner)
        if action.tool not in available and action.tool not in TOOL_DEFINITIONS:
            obs = self._build_observation(events=[f"Invalid tool: {action.tool}"])
            reward = Reward(value=-0.1, components={"invalid_tool": -0.1},
                            message=f"Tool '{action.tool}' not available in phase '{state.phase}'")
            return obs, reward, False, {"error": "invalid_tool"}

        # --- Execute tool ---
        state_before = copy.deepcopy(state)
        action_dict = {"tool": action.tool, "params": action.parameters}

        # --- Advance simulated clock ---
        elapsed_hours = ACTION_TIME_HOURS.get(action.tool, 0.083)
        state.sim_time_hours = round(state.sim_time_hours + elapsed_hours, 4)

        obs_updates, phys_effects, cost, tool_msg = self._tool_executor.execute(
            action, state, self._rng, sim_time_hours=state.sim_time_hours
        )

        # --- Budget hard-stop: block paid actions when over budget ---
        if cost > 0 and state.owner.budget_limit is not None:
            remaining = state.owner.budget_limit - state.owner.budget_spent
            if remaining <= 0:
                obs = self._build_observation(events=[
                    f"BUDGET EXHAUSTED: £{state.owner.budget_spent:.0f} spent of £{state.owner.budget_limit:.0f} limit. "
                    f"'{action.tool}' costs £{cost:.0f} — cannot proceed. "
                    "Only free actions (check_vitals, physical_exam, decide_triage_route, make_disposition) are available."
                ])
                reward = Reward(
                    value=-0.25,
                    components={"budget_exceeded": -0.25},
                    message=f"Over budget — {action.tool} blocked (£{cost:.0f})",
                )
                state.step += 1
                state.phase_step += 1
                self._action_log.append(action_dict)
                self._state = state
                return obs, reward, False, {"error": "budget_exceeded"}

        # --- Apply financial cost ---
        state.owner.budget_spent += cost

        # --- Apply observation updates ---
        self._apply_obs_updates(obs_updates, state)

        # --- Apply IV placement if signalled ---
        if obs_updates.get("iv_placed"):
            state.patient = state.patient.model_copy(update={"iv_access": True})

        # --- Queue treatment effects ---
        self._treatment_effects_pending.extend(phys_effects)

        # --- Update physiology (scaled by elapsed simulated time) ---
        if profile:
            state.patient = update_physiology(
                state.patient, profile, self._treatment_effects_pending,
                self._rng, elapsed_hours=elapsed_hours,
            )
            self._treatment_effects_pending = []

        # --- Advance async jobs (keep step counter for display, but readiness is time-based) ---
        events = []
        for job_id, job in list(state.pending_async.items()):
            if job.eta_steps > 0:
                new_eta = job.eta_steps - 1
                state.pending_async[job_id] = job.model_copy(update={"eta_steps": new_eta})
            # Time-based ready notification
            if state.sim_time_hours >= job.eta_hours and job.eta_hours > 0:
                events.append(
                    f"Result ready: {job.panel_or_modality} — "
                    f"call collect_result(job_id='{job_id}')"
                )

        # --- Fire scheduled events ---
        events.extend(self._check_events(state, meta))

        # --- Log action ---
        self._action_log.append(action_dict)
        state.action_history.append(
            f"[{state.step}] {action.tool}({action.parameters}) → {tool_msg[:80]}"
        )

        # --- Per-step reward ---
        step_reward_val, step_components, step_msg = compute_step_reward(
            action_dict, state_before, state, obs_updates, meta, self._action_log
        )

        # --- Advance step counters ---
        state.step += 1
        state.phase_step += 1

        # --- Check for phase transitions ---
        done, terminal_reason = self._check_terminal(action, state, meta)

        # --- Handle phase transition ---
        if not done and self._should_advance_phase(action, state):
            state = self._advance_phase(action, state)
            events.append(f"Phase advanced to: {state.phase}")

        # --- Terminal reward ---
        terminal_reward_val = 0.0
        terminal_components = {}
        if done:
            grade = grade_episode(state, meta, self._action_log)
            terminal_reward_val = (grade.score - 0.5) * 4.0  # scale to [-2, +2]
            terminal_components = {f"terminal_{k}": v for k, v in grade.breakdown.items()}
            terminal_components["grade_score"] = grade.score
            info = {
                "grade": grade.score,
                "grade_breakdown": grade.breakdown,
                "grade_feedback": grade.feedback,
                "passed": grade.passed,
                "total_steps": state.step,
                "terminal_reason": terminal_reason,
            }
        else:
            info = {
                "step": state.step,
                "phase": state.phase,
                "severity": state.patient.severity,
                "tool_message": tool_msg,
            }

        total_reward_val = step_reward_val + terminal_reward_val
        all_components = {**step_components, **terminal_components}
        self._cumulative_reward += total_reward_val

        reward = Reward(
            value=round(total_reward_val, 3),
            components=all_components,
            message=step_msg if not done else f"Episode complete. Grade: {terminal_components.get('grade_score', 0):.3f}",
        )

        obs = self._build_observation(events=events)
        if done:
            obs = obs.model_copy(update={
                "episode_done": True,
                "terminal_reason": terminal_reason,
            })

        self._state = state
        return obs, reward, done, info

    # -----------------------------------------------------------------------
    # state()
    # -----------------------------------------------------------------------

    def state(self) -> dict:
        """Return the full internal state (hidden ground truth included)."""
        if self._state is None:
            return {}
        return {
            "internal_state": self._state.model_dump(),
            "meta": {k: v for k, v in (self._meta or {}).items()
                     if k != "task_spec"},
            "action_log": self._action_log,
            "cumulative_reward": self._cumulative_reward,
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_observation(self, events: list) -> Observation:
        state = self._state
        meta = self._meta or {}
        patient = state.patient
        owner = state.owner

        # Pending results
        pending = [
            AsyncJob(
                job_id=job_id,
                tool=job.tool,
                panel_or_modality=job.panel_or_modality,
                eta_steps=job.eta_steps,
                eta_hours=job.eta_hours,
                cost_incurred=job.cost_incurred,
            )
            for job_id, job in state.pending_async.items()
        ]

        # Available tools for this phase
        available = get_available_tools(state.phase, patient, owner)

        # Phase-specific fields
        avail_interventions = None
        monitoring_trends = None
        disposition_opts = None

        if state.phase == "stabilisation":
            avail_interventions = [
                "administer_fluid_bolus", "give_medication",
                "oxygen_therapy", "perform_procedure", "place_iv_access"
            ]
        elif state.phase == "monitoring":
            # Generate trend description
            monitoring_trends = {
                "heart_rate": "improving" if patient.heart_rate < 120 else "elevated",
                "spo2": "stable" if patient.spo2 > 94 else "declining",
                "mentation": patient.mentation,
            }
        elif state.phase == "disposition":
            disposition_opts = [
                "admit_icu", "admit_ward", "refer_specialist",
                "discharge_with_medication", "discharge_no_treatment",
                "euthanasia", "owner_declined"
            ]

        # Action history (truncated if handoff occurred)
        history = state.action_history
        if state.handoff_occurred:
            history = [
                "--- SHIFT HANDOFF: prior history summarised ---",
                f"Patient arrived with: {meta.get('presenting_complaint', '')[:100]}",
                f"Current phase: {state.phase}",
                f"Budget spent: £{owner.budget_spent:.0f}",
            ] + state.action_history[-3:]

        return Observation(
            step=state.step,
            phase=state.phase,
            phase_step=state.phase_step,
            phase_step_limit=self._phase_limits.get(state.phase, 20),
            sim_time_hours=round(state.sim_time_hours, 3),
            episode_done=False,
            species=meta.get("species", "unknown"),
            breed=meta.get("breed_name", "unknown"),
            age_years=meta.get("age_years", 0.0),
            sex=meta.get("sex", "unknown"),
            weight_kg=meta.get("weight_kg", 0.0),
            presenting_complaint=meta.get("presenting_complaint", ""),
            vitals=getattr(self, "_last_vitals", None),
            physical_exam_findings=getattr(self, "_phys_findings", {}),
            lab_results=getattr(self, "_lab_results", {}),
            imaging_results=getattr(self, "_imaging_results", {}),
            pending_results=pending,
            owner_contact_established=owner.contact_established,
            budget_limit=owner.budget_limit if owner.contact_established else None,
            budget_spent=owner.budget_spent,
            budget_remaining=(
                round(owner.budget_limit - owner.budget_spent, 2)
                if owner.contact_established and owner.budget_limit is not None
                else None
            ),
            consent_items=owner.consent_items,
            specialist_opinion=state.specialist_opinion,
            events=events,
            action_history=history,
            handoff_occurred=state.handoff_occurred,
            available_interventions=avail_interventions,
            monitoring_trends=monitoring_trends,
            disposition_options=disposition_opts,
            available_tools=available,
        )

    def _apply_obs_updates(self, updates: dict, state: FullState):
        """Apply tool result updates to persistent observation state."""
        if not hasattr(self, "_last_vitals"):
            self._last_vitals = None
            self._phys_findings = {}
            self._lab_results = {}
            self._imaging_results = {}

        if "vitals" in updates:
            # Merge into existing vitals snapshot
            if self._last_vitals:
                merged = {**self._last_vitals, **{k: v for k, v in updates["vitals"].items() if v is not None}}
                self._last_vitals = merged
            else:
                self._last_vitals = updates["vitals"]

        if "physical_exam_findings" in updates:
            self._phys_findings.update(updates["physical_exam_findings"])

        if "lab_results_new" in updates:
            self._lab_results.update(updates["lab_results_new"])

        if "imaging_results_new" in updates:
            self._imaging_results.update(updates["imaging_results_new"])

        if "specialist_opinion" in updates:
            state.specialist_opinion = updates["specialist_opinion"]

        if "owner_contacted" in updates:
            state.owner.contact_established = True

        if "consent_update" in updates:
            for item in updates["consent_update"]:
                if item not in state.owner.consent_items:
                    state.owner.consent_items.append(item)

        if "pending_new" in updates:
            p = updates["pending_new"]
            job_id = p["job_id"]
            # Only create if not already in pending_async (tools.py may have set it with eta_hours)
            if job_id not in state.pending_async:
                state.pending_async[job_id] = AsyncJob(
                    job_id=job_id,
                    tool=p["tool"],
                    panel_or_modality=p["panel"],
                    eta_steps=p["eta_steps"],
                    eta_hours=p.get("eta_hours", 0.0),
                    cost_incurred=0.0,
                )

    def _check_events(self, state: FullState, meta: dict) -> list:
        """Check and fire scheduled mid-episode events."""
        events = []
        scheduled = meta.get("events_scheduled", [])

        for event in scheduled:
            if event.get("fired"):
                continue
            if event["step"] == state.step:
                etype = event["type"]
                event["fired"] = True

                if etype == "seizure" and not state.seizure_fired:
                    state.seizure_fired = True
                    # Worsen physiology
                    state.patient = state.patient.model_copy(update={
                        "heart_rate": state.patient.heart_rate + 25,
                        "severity": state.patient.severity + 0.05,
                        "cooperation_score": max(0.2, state.patient.cooperation_score - 0.4),
                    })
                    events.append(
                        "EVENT: Patient has a generalised tonic-clonic seizure! "
                        "Immediate intervention required. Give diazepam or midazolam IV/intranasal."
                    )

                elif etype == "shift_handoff":
                    state.handoff_occurred = True
                    state.handoff_step = state.step
                    events.append(
                        "EVENT: Shift change — you are the incoming clinician. "
                        "Action history has been summarised. Review current status."
                    )

                elif etype == "budget_change":
                    delta = event.get("delta", 0)
                    state.owner.budget_limit = max(0, state.owner.budget_limit + delta)
                    if delta < 0:
                        events.append(
                            f"EVENT: Owner has revised budget down by £{abs(delta):.0f}. "
                            f"New limit: £{state.owner.budget_limit:.0f}. Reprioritise examinations."
                        )
                    else:
                        events.append(
                            f"EVENT: Owner has increased budget by £{delta:.0f}. "
                            f"New limit: £{state.owner.budget_limit:.0f}."
                        )

        # Check for spontaneous deterioration events
        if state.patient.severity > 0.75 and not getattr(state, "_critical_alert_fired", False):
            state.__dict__["_critical_alert_fired"] = True
            events.append(
                "ALERT: Patient entering critical condition. "
                f"Severity {state.patient.severity:.2f}. Immediate intervention required."
            )

        return events

    def _check_terminal(
        self, action: Action, state: FullState, meta: dict
    ) -> Tuple[bool, Optional[str]]:
        """Check all terminal conditions."""
        # Patient death
        profile = DIAGNOSIS_PROFILES.get(state.patient.true_diagnosis)
        if profile and state.patient.severity >= profile.lethal_threshold:
            return True, "patient_death"

        # Final disposition action
        if action.tool == "make_disposition":
            state.disposition_made = action.parameters.get("decision")
            return True, "disposition_made"

        # Step limit
        if state.step >= self.max_total_steps:
            return True, "step_limit_exceeded"

        # Phase step limit exceeded
        limit = self._phase_limits.get(state.phase, 20)
        if state.phase_step >= limit:
            if state.phase == "disposition":
                return True, "disposition_timeout"
            # Force advance
            return False, None

        return False, None

    def _should_advance_phase(self, action: Action, state: FullState) -> bool:
        """Determine if we should advance to the next phase."""
        if action.tool == "decide_triage_route":
            route = action.parameters.get("urgency", "urgent_stabilise")
            state.triage_route_decided = route
            if route in ("discharge", "owner_decline_treatment"):
                # Skip to disposition
                return True
            return True

        limit = self._phase_limits.get(state.phase, 20)
        if state.phase_step >= limit and state.phase != "disposition":
            return True

        return False

    def _advance_phase(self, action: Action, state: FullState) -> FullState:
        """Move to the next phase."""
        route = state.triage_route_decided
        current = state.phase

        if current == "triage":
            if route in ("discharge", "owner_decline_treatment"):
                next_phase = "disposition"
            else:
                next_phase = "stabilisation"
        elif current == "stabilisation":
            next_phase = "monitoring"
        elif current == "monitoring":
            next_phase = "disposition"
        else:
            next_phase = "disposition"

        state.phase = next_phase
        state.phase_step = 0
        return state

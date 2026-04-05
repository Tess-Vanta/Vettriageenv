"""
VetTriageEnv — OpenEnv-compliant FastAPI server.

Wraps the VetTriageEnv logic in openenv_core's Environment/Action/Observation/State
base classes and exposes it via create_fastapi_app().
"""
from __future__ import annotations

import logging
import traceback
import warnings
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# openenv_core is the legacy module name; openenv.core is the new one.
# The deprecation warning is expected — suppressed intentionally.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from openenv_core import (
        Action as BaseAction,
        Observation as BaseObservation,
        State as BaseState,
        Environment,
        create_fastapi_app,
    )

from pydantic import Field

from vettriagevenv.env import VetTriageEnv as _VetTriageEnv
from vettriagevenv.models import Action as _VetAction
from vettriagevenv.tasks import list_tasks


# ---------------------------------------------------------------------------
# OpenEnv-typed Action
# ---------------------------------------------------------------------------

class VetTriageAction(BaseAction):
    """
    A structured tool call to the veterinary triage environment.

    The agent picks a tool and supplies its parameters as a dict.
    Optional reasoning field for chain-of-thought logging.
    """
    tool: str = Field(
        description=(
            "Tool to call. Options: check_vitals, physical_exam, run_bloodwork, "
            "run_imaging, collect_result, place_iv_access, administer_fluid_bolus, "
            "give_medication, oxygen_therapy, perform_procedure, contact_owner, "
            "consult_specialist, decide_triage_route, make_disposition"
        )
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific parameters. See README for per-tool parameter schemas."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional clinical reasoning for this action (used for logging)."
    )


# ---------------------------------------------------------------------------
# OpenEnv-typed Observation
# ---------------------------------------------------------------------------

class VetTriageObservation(BaseObservation):
    """
    What the agent sees after each step.

    Observation builds incrementally — fields are None until the relevant
    tool has been called (e.g. vitals are None until check_vitals is called).
    """
    # Episode context
    step: int = Field(default=0)
    phase: str = Field(default="triage",
        description="triage | stabilisation | monitoring | disposition")
    phase_step: int = Field(default=0)
    phase_step_limit: int = Field(default=20)

    # Patient identity (always present)
    species: str = Field(default="")
    breed: str = Field(default="")
    age_years: float = Field(default=0.0)
    sex: str = Field(default="")
    weight_kg: float = Field(default=0.0)
    presenting_complaint: str = Field(default="")

    # Examination findings (populated progressively)
    vitals: Optional[Dict[str, Any]] = Field(default=None)
    physical_exam_findings: Dict[str, str] = Field(default_factory=dict)
    lab_results: Dict[str, Any] = Field(default_factory=dict)
    imaging_results: Dict[str, Any] = Field(default_factory=dict)

    # Async jobs pending
    pending_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Owner / financial
    owner_contact_established: bool = Field(default=False)
    budget_limit: Optional[float] = Field(default=None)
    budget_spent: float = Field(default=0.0)
    consent_items: List[str] = Field(default_factory=list)

    # Specialist opinion
    specialist_opinion: Optional[str] = Field(default=None)

    # Mid-episode events (seizures, budget changes, handoffs)
    events: List[str] = Field(default_factory=list)

    # Action history
    action_history: List[str] = Field(default_factory=list)
    handoff_occurred: bool = Field(default=False)

    # Phase-specific
    available_interventions: Optional[List[str]] = Field(default=None)
    monitoring_trends: Optional[Dict[str, str]] = Field(default=None)
    disposition_options: Optional[List[str]] = Field(default=None)

    # Available tools
    available_tools: List[str] = Field(default_factory=list)

    # Terminal
    terminal_reason: Optional[str] = Field(default=None)

    # Task info
    task_id: str = Field(default="random")
    available_tasks: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# OpenEnv-typed State
# ---------------------------------------------------------------------------

class VetTriageState(BaseState):
    """
    Full internal state (hidden ground truth + episode context).
    Returned by state() for logging and debugging.
    Not shown to the agent during normal operation.
    """
    true_diagnosis: Optional[str] = Field(default=None)
    patient_severity: Optional[float] = Field(default=None)
    triage_route_decided: Optional[str] = Field(default=None)
    disposition_made: Optional[str] = Field(default=None)
    cumulative_reward: float = Field(default=0.0)
    action_log: List[Dict[str, Any]] = Field(default_factory=list)
    phase: str = Field(default="triage")
    total_steps: int = Field(default=0)


# ---------------------------------------------------------------------------
# OpenEnv Environment implementation
# ---------------------------------------------------------------------------

class VetTriageEnvironment(Environment[VetTriageAction, VetTriageObservation, VetTriageState]):
    """
    Veterinary triage RL environment implementing the OpenEnv interface.

    An AI agent must gather clinical information via structured tool calls
    and make correct triage decisions for procedurally-generated animal patients.

    Phases: triage → stabilisation → monitoring → disposition
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env = _VetTriageEnv(max_total_steps=100)
        self._current_task_id = "random"
        self._last_obs = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy_gdv",
        **kwargs,
    ) -> VetTriageObservation:
        """
        Reset the environment to a new episode.

        Args:
            seed: Random seed for reproducibility.
            task_id: One of 'easy_gdv', 'medium_hcm_cat', 'hard_polytrauma',
                     or 'random' for procedural generation.
        """
        self._current_task_id = task_id
        obs = self._env.reset(task_id=task_id, seed=seed)
        result = self._convert_obs(obs)
        self._last_obs = result
        return result

    def step(
        self,
        action: VetTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> VetTriageObservation:
        """
        Take one action in the environment.

        The action is a structured tool call. Each call costs one step.
        The patient's condition may deteriorate — act efficiently.
        """
        # Auto-reset if no active session
        if self._env._state is None:
            logger.info("No active session, auto-resetting with easy_gdv")
            self._env.reset(task_id=self._current_task_id or "easy_gdv", seed=42)

        try:
            vet_action = _VetAction(
                tool=action.tool,
                parameters=action.parameters,
                reasoning=action.reasoning,
            )
        except Exception as e:
            logger.error(f"Action construction failed: {e}\n{traceback.format_exc()}")
            raise

        try:
            obs, reward, done, info = self._env.step(vet_action)
        except Exception as e:
            logger.error(f"Step failed: {e}\n{traceback.format_exc()}")
            raise
        result = self._convert_obs(obs)
        result.reward = reward.value
        result.done = done
        result.metadata = {
            "reward_components": reward.components,
            "reward_message": reward.message,
            "info": {k: v for k, v in info.items() if k != "grade_breakdown"},
        }
        if done and "grade" in info:
            result.metadata["grade"] = info["grade"]
            result.metadata["passed"] = info["passed"]
            result.metadata["grade_feedback"] = info.get("grade_feedback", [])
            result.metadata["grade_breakdown"] = info.get("grade_breakdown", {})
        self._last_obs = result
        return result

    @property
    def state(self) -> VetTriageState:
        """Return full internal state including hidden ground truth."""
        raw = self._env.state()
        internal = raw.get("internal_state", {})
        patient = internal.get("patient", {})
        return VetTriageState(
            true_diagnosis=patient.get("true_diagnosis"),
            patient_severity=patient.get("severity"),
            triage_route_decided=internal.get("triage_route_decided"),
            disposition_made=internal.get("disposition_made"),
            cumulative_reward=raw.get("cumulative_reward", 0.0),
            action_log=raw.get("action_log", []),
            phase=internal.get("phase", "triage"),
            total_steps=internal.get("step", 0),
        )

    def get_metadata(self):
        from openenv_core import Environment
        # Return metadata compatible with the framework
        return {
            "name": "VetTriageEnv",
            "version": "1.0.0",
            "description": (
                "Veterinary triage RL environment. Agent gathers clinical information "
                "via tool calls and makes correct triage decisions for animal patients."
            ),
            "tasks": list_tasks(),
            "reward_range": [-5.0, 5.0],
        }

    # -----------------------------------------------------------------------
    # Conversion helper
    # -----------------------------------------------------------------------

    def _convert_obs(self, obs) -> VetTriageObservation:
        """Convert internal Observation to OpenEnv VetTriageObservation."""
        # Convert vitals (VitalsSnapshot pydantic model → dict)
        vitals_dict = None
        if obs.vitals is not None:
            if hasattr(obs.vitals, "model_dump"):
                vitals_dict = obs.vitals.model_dump()
            elif isinstance(obs.vitals, dict):
                vitals_dict = obs.vitals

        # Convert pending results
        pending = []
        for p in obs.pending_results:
            if hasattr(p, "model_dump"):
                pending.append(p.model_dump())
            elif isinstance(p, dict):
                pending.append(p)

        return VetTriageObservation(
            done=obs.episode_done,
            reward=None,
            step=obs.step,
            phase=obs.phase,
            phase_step=obs.phase_step,
            phase_step_limit=obs.phase_step_limit,
            species=obs.species,
            breed=obs.breed,
            age_years=obs.age_years,
            sex=obs.sex,
            weight_kg=obs.weight_kg,
            presenting_complaint=obs.presenting_complaint,
            vitals=vitals_dict,
            physical_exam_findings=obs.physical_exam_findings,
            lab_results=obs.lab_results,
            imaging_results=obs.imaging_results,
            pending_results=pending,
            owner_contact_established=obs.owner_contact_established,
            budget_limit=obs.budget_limit,
            budget_spent=obs.budget_spent,
            consent_items=obs.consent_items,
            specialist_opinion=obs.specialist_opinion,
            events=obs.events,
            action_history=obs.action_history,
            handoff_occurred=obs.handoff_occurred,
            available_interventions=obs.available_interventions,
            monitoring_trends=obs.monitoring_trends,
            disposition_options=obs.disposition_options,
            available_tools=obs.available_tools,
            terminal_reason=obs.terminal_reason,
            task_id=self._current_task_id,
            available_tasks=list_tasks(),
        )


# ---------------------------------------------------------------------------
# FastAPI app (entry point)
# ---------------------------------------------------------------------------

app = create_fastapi_app(
    env=VetTriageEnvironment,
    action_cls=VetTriageAction,
    observation_cls=VetTriageObservation,
)


def main():
    """Entry point for 'server' script (required by openenv validate)."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()

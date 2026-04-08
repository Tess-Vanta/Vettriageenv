"""
Typed Pydantic models for VetTriageEnv — OpenEnv spec compliance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Lab / Imaging result types
# ---------------------------------------------------------------------------

class CBCResult(BaseModel):
    hematocrit: float           # %
    total_protein: float        # g/dL
    white_cell_count: float     # x10^9/L
    platelets: float            # x10^9/L
    interpretation: str

class ChemistryResult(BaseModel):
    bun: float                  # mg/dL
    creatinine: float           # mg/dL
    alt: float                  # U/L
    alp: float                  # U/L
    albumin: float              # g/dL
    glucose: float              # mg/dL
    sodium: float               # mEq/L
    potassium: float            # mEq/L
    interpretation: str

class BloodGasResult(BaseModel):
    ph: float
    pco2: float                 # mmHg
    po2: float                  # mmHg
    hco3: float                 # mEq/L
    lactate: float              # mmol/L
    interpretation: str

class RadiographResult(BaseModel):
    region: str
    findings: List[str]
    interpretation: str

class UltrasoundResult(BaseModel):
    region: str
    findings: List[str]
    free_fluid_present: bool
    interpretation: str


# ---------------------------------------------------------------------------
# Vitals snapshot (visible to agent after check_vitals)
# ---------------------------------------------------------------------------

class VitalsSnapshot(BaseModel):
    heart_rate: Optional[float] = None          # bpm
    respiratory_rate: Optional[float] = None    # breaths/min
    temperature: Optional[float] = None         # Celsius
    spo2: Optional[float] = None                # %
    systolic_bp: Optional[float] = None         # mmHg
    mucous_membrane_color: Optional[str] = None
    capillary_refill_time: Optional[float] = None   # seconds
    mentation: Optional[str] = None
    pain_score: Optional[int] = None            # 0-10
    systems_checked: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Async job tracking
# ---------------------------------------------------------------------------

class AsyncJob(BaseModel):
    job_id: str
    tool: str
    panel_or_modality: str
    eta_steps: int              # kept for backward compat (approx steps)
    eta_hours: float = 0.0     # absolute sim clock time when result is ready
    cost_incurred: float


# ---------------------------------------------------------------------------
# Action model (what the agent sends)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    tool: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation model (what the agent receives each step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    # Episode context
    step: int
    phase: Literal["triage", "stabilisation", "monitoring", "disposition"]
    phase_step: int
    phase_step_limit: int
    episode_done: bool = False

    # Patient identity (always visible)
    species: str
    breed: str
    age_years: float
    sex: str
    weight_kg: float
    presenting_complaint: str

    # Examination findings (populated as tools are called)
    vitals: Optional[VitalsSnapshot] = None
    physical_exam_findings: Dict[str, str] = Field(default_factory=dict)

    # Lab and imaging results
    lab_results: Dict[str, Any] = Field(default_factory=dict)
    imaging_results: Dict[str, Any] = Field(default_factory=dict)

    # Async jobs pending
    pending_results: List[AsyncJob] = Field(default_factory=list)

    # Owner / financial
    owner_contact_established: bool = False
    budget_limit: Optional[float] = None
    budget_spent: float = 0.0
    budget_remaining: Optional[float] = None  # None until owner contacted
    consent_items: List[str] = Field(default_factory=list)

    # Specialist opinion (after consult)
    specialist_opinion: Optional[str] = None

    # Mid-episode events that fired this step
    events: List[str] = Field(default_factory=list)

    # Action history summary
    action_history: List[str] = Field(default_factory=list)
    handoff_occurred: bool = False

    # Phase-specific
    available_interventions: Optional[List[str]] = None  # phase 2
    monitoring_trends: Optional[Dict[str, str]] = None   # phase 3
    disposition_options: Optional[List[str]] = None      # phase 4

    # Available tools this step
    available_tools: List[str] = Field(default_factory=list)

    # Simulated time clock (hours elapsed since case start)
    sim_time_hours: float = 0.0

    # Terminal info
    terminal_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


# ---------------------------------------------------------------------------
# Full internal state (returned by state())
# ---------------------------------------------------------------------------

class PatientInternalState(BaseModel):
    """Hidden ground truth — not visible to agent."""
    true_diagnosis: str
    severity: float             # 0.0–1.0; increases over time
    deterioration_rate: float
    max_survivable_steps: int

    heart_rate: float
    respiratory_rate: float
    temperature: float
    spo2: float
    systolic_bp: float
    mucous_membrane_color: str
    capillary_refill_time: float
    mentation: str
    pain_score: int
    hydration_deficit: float    # 0.0–1.0

    iv_access: bool = False
    sedation_level: float = 0.0
    cooperation_score: float = 1.0
    pain_amplifier: float = 1.0

    cbc: Optional[CBCResult] = None
    chemistry: Optional[ChemistryResult] = None
    blood_gas: Optional[BloodGasResult] = None
    lactate: Optional[float] = None
    radiograph_thorax: Optional[RadiographResult] = None
    radiograph_abdomen: Optional[RadiographResult] = None
    ultrasound_abdomen: Optional[UltrasoundResult] = None

    treatments_given: List[str] = Field(default_factory=list)
    fluid_type_active: Optional[str] = None


class OwnerInternalState(BaseModel):
    budget_limit: float
    budget_spent: float = 0.0
    contact_established: bool = False
    consent_items: List[str] = Field(default_factory=list)
    history_reliability: float = 0.9
    budget_change_at_step: Optional[int] = None
    budget_change_delta: Optional[float] = None


class FullState(BaseModel):
    patient: PatientInternalState
    owner: OwnerInternalState
    step: int
    phase: str
    phase_step: int
    triage_route_decided: Optional[str] = None
    disposition_made: Optional[str] = None
    pending_async: Dict[str, AsyncJob] = Field(default_factory=dict)
    async_results: Dict[str, Any] = Field(default_factory=dict)
    action_history: List[str] = Field(default_factory=list)
    handoff_occurred: bool = False
    handoff_step: Optional[int] = None
    events_fired: List[str] = Field(default_factory=list)
    seizure_fired: bool = False
    specialist_opinion: Optional[str] = None
    task_id: str = "random"
    sim_time_hours: float = 0.0  # simulated clock: hours elapsed since case start

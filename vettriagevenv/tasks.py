"""
Benchmark task definitions for VetTriageEnv.
Each task has a fixed seed, expected difficulty, and optimal policy notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskSpec:
    task_id: str
    name: str
    difficulty: str                     # "easy" | "medium" | "hard"
    description: str
    seed: int
    force_diagnosis: str
    force_species: str
    optimal_route: str
    optimal_disposition: str
    passing_threshold: float            # minimum score to "pass"
    max_steps: int
    notes: List[str] = field(default_factory=list)


TASK_REGISTRY: Dict[str, TaskSpec] = {
    # ------------------------------------------------------------------
    # EASY: Clear emergency, one right answer, no confounders
    # ------------------------------------------------------------------
    "easy_gdv": TaskSpec(
        task_id="easy_gdv",
        name="The Obvious Emergency — GDV",
        difficulty="easy",
        description=(
            "A large-breed dog presents with acute abdominal distension and unproductive "
            "retching. This is a classic gastric dilatation-volvulus (GDV). The agent must "
            "recognise the emergency quickly, perform targeted examinations, and route to "
            "immediate resuscitation without over-examining. Budget is not a constraint. "
            "A competent agent should achieve this in under 8 steps."
        ),
        seed=42,
        force_diagnosis="gastric_dilatation_volvulus",
        force_species="dog",
        optimal_route="immediate_resuscitation",
        optimal_disposition="admit_icu",
        passing_threshold=0.60,
        max_steps=30,
        notes=[
            "Check vitals (cardiovascular) → physical exam abdomen → decide route",
            "Running full bloodwork before deciding route wastes critical time",
            "GDV dog should never be discharged — automatic fail",
            "Gastric decompression in stabilisation phase improves outcome significantly",
        ],
    ),

    # ------------------------------------------------------------------
    # MEDIUM: Ambiguous presentation, requires differential reasoning
    # ------------------------------------------------------------------
    "medium_hcm_cat": TaskSpec(
        task_id="medium_hcm_cat",
        name="The Dyspnoeic Cat — Wait vs. Treat Dilemma",
        difficulty="medium",
        description=(
            "An older Maine Coon cat presents with acute dyspnoea. SpO2 is 82% and dropping. "
            "The differential includes hypertrophic cardiomyopathy (HCM) with pleural effusion, "
            "feline asthma, and anaemia. The TIME-DELAY DILEMMA: ordering routine bloodwork "
            "takes 2h and routine imaging 1.5h — during which the cat's SpO2 continues to fall. "
            "An agent that waits for perfect diagnostics will watch the patient suffocate. "
            "The correct strategy: TREAT FIRST (oxygen + furosemide + thoracocentesis based on "
            "clinical signs), then confirm with STAT imaging — not the other way around. "
            "Crystalloid fluids are lethal in HCM. Owner has moderate budget (£800). "
            "Grading explicitly penalises high sim_time_hours at first treatment."
        ),
        seed=137,
        force_diagnosis="hypertrophic_cardiomyopathy",
        force_species="cat",
        optimal_route="urgent_stabilise",
        optimal_disposition="admit_ward",
        passing_threshold=0.55,
        max_steps=45,
        notes=[
            "TIME PRESSURE: SpO2 starts ~85%, drops ~2%/step. Cat dies below SpO2 60%.",
            "WRONG: run_bloodwork(routine) -> run_imaging(routine) -> wait -> treat (2h+ lost)",
            "RIGHT: oxygen_therapy -> physical_exam thorax -> furosemide -> thoracocentesis -> STAT imaging",
            "Grader rewards: first_treatment_before T+0.30h, penalises treatment after T+1.0h",
            "Crystalloid bolus in HCM cat causes pulmonary oedema — harmful action penalty",
            "Thoracocentesis dramatically improves respiratory status",
            "STAT imaging (30min) is much faster than routine (90min) — worth the extra cost",
        ],
    ),

    # ------------------------------------------------------------------
    # HARD: Multi-system, budget constraints, mid-episode events
    # ------------------------------------------------------------------
    "hard_polytrauma": TaskSpec(
        task_id="hard_polytrauma",
        name="The Polytrauma — Hit By Car with Budget Constraint",
        difficulty="hard",
        description=(
            "A young dog has been hit by a car. The agent must manage polytrauma including "
            "suspected pneumothorax, haemoabdomen, and possible spinal injury — all simultaneously. "
            "A seizure occurs mid-examination at step 7. Owner has a tight budget (£600). "
            "Crystalloid bolus is relatively contraindicated (haemoabdomen). "
            "Thoracocentesis for pneumothorax is urgent — delay beyond step 12 causes rapid deterioration. "
            "The agent must prioritise imaging over comprehensive bloodwork given the budget."
        ),
        seed=999,
        force_diagnosis="polytrauma_hbc",
        force_species="dog",
        optimal_route="immediate_resuscitation",
        optimal_disposition="admit_icu",
        passing_threshold=0.45,
        max_steps=60,
        notes=[
            "Run thorax AND abdomen imaging urgently within first 5 steps",
            "Seizure at step 7: give diazepam/midazolam immediately",
            "Thoracocentesis required before step 12 or patient deteriorates rapidly",
            "Colloid preferred over crystalloid for haemoabdomen",
            "Budget of £600 forces prioritisation: imaging > full bloodwork",
            "Avoid thorough spinal exam without sedation — risk of cord injury",
        ],
    ),
}


def get_task(task_id: str) -> Optional[TaskSpec]:
    return TASK_REGISTRY.get(task_id)


def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())

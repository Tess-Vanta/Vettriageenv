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
        name="The Dyspnoeic Cat — HCM with Pleural Effusion",
        difficulty="medium",
        description=(
            "An older Maine Coon cat presents with acute dyspnoea. The differential includes "
            "hypertrophic cardiomyopathy (HCM) with pleural effusion, feline asthma, and "
            "anaemia-related respiratory distress. The agent must reason through the differential, "
            "select examinations that distinguish between these, and avoid the dangerous mistake "
            "of giving crystalloid fluids to a cardiac patient. Owner has a moderate budget (£800). "
            "Thoracocentesis is the key therapeutic intervention."
        ),
        seed=137,
        force_diagnosis="hypertrophic_cardiomyopathy",
        force_species="cat",
        optimal_route="urgent_stabilise",
        optimal_disposition="admit_ward",
        passing_threshold=0.55,
        max_steps=45,
        notes=[
            "Radiograph thorax is diagnostic — pleural effusion immediately visible",
            "Crystalloid bolus in HCM cat causes pulmonary oedema — major penalty",
            "Thoracocentesis dramatically improves respiratory status",
            "Contact owner early for consent and budget discussion",
            "Furosemide is appropriate first-line for CHF/HCM",
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

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
            "Crystalloid fluids are lethal in HCM. Owner has moderate budget (₹80,000). "
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
    # HARD: Resource Scarcity — strict budget forces information prioritisation
    # ------------------------------------------------------------------
    "hard_imha_budget": TaskSpec(
        task_id="hard_imha_budget",
        name="The Budget Crisis — IMHA with ₹38,000 Limit",
        difficulty="hard",
        description=(
            "A 5-year-old Cocker Spaniel presents with progressive weakness, pale/icteric "
            "mucous membranes, and collapse. The differential includes immune-mediated haemolytic "
            "anaemia (IMHA), internal haemorrhage, and hepatic failure. "
            "THE RESOURCE SCARCITY DILEMMA: The owner reveals a hard budget of ₹38,000. "
            "A brute-force agent running all diagnostics costs ₹63,500+. "
            "The agent must choose: CBC (₹4,500) OR chemistry (₹8,000) OR imaging (₹9,000-13,000) — "
            "not all three. The correct strategy: CBC + lactate (₹8,000 total) is sufficient to confirm "
            "IMHA and guide treatment. Imaging and full chemistry are wasteful here. "
            "Any agent that runs full_panel (₹16,000) + ultrasound (₹13,000) + bloodwork + procedures "
            "will exceed budget and be blocked from administering critical treatments. "
            "Contact owner FIRST to discover the budget before ordering anything."
        ),
        seed=777,
        force_diagnosis="immune_haemolytic_anaemia",
        force_species="dog",
        optimal_route="urgent_stabilise",
        optimal_disposition="admit_icu",
        passing_threshold=0.50,
        max_steps=40,
        notes=[
            "BUDGET: ₹38,000 hard limit — contact_owner first to learn this",
            "WRONG: full_panel(₹16,000) + ultrasound(₹13,000) + procedures = BLOCKED at ₹38,000",
            "RIGHT: cbc(₹4,500) + lactate(₹3,500) = ₹8,000 confirms IMHA, leaves ₹30,000 for treatment",
            "Haemic murmur + icteric MMs + tachycardia = IMHA without any bloodwork",
            "Blood products (₹3,500) + immunosuppression (give_medication) are key treatments",
            "Crystalloid fluids are NOT harmful here (unlike HCM) but wasteful on budget",
            "Grader scores: budget_efficiency is 15% of total score on this task",
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
            "A seizure occurs mid-examination at step 7. Owner has a tight budget (₹60,000). "
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
            "Budget of ₹60,000 forces prioritisation: imaging > full bloodwork",
            "Avoid thorough spinal exam without sedation — risk of cord injury",
        ],
    ),

    # ------------------------------------------------------------------
    # HARD: Stochastic failures — closed-loop vs open-loop agents
    # ------------------------------------------------------------------
    "hard_stochastic_pancreatitis": TaskSpec(
        task_id="hard_stochastic_pancreatitis",
        name="The Uncooperative Patient — Pancreatitis with High Failure Rate",
        difficulty="hard",
        description=(
            "A 7-year-old Border Collie presents with severe acute pancreatitis: vomiting, "
            "cranial abdominal pain, and lethargy. The dog is highly stressed and uncooperative "
            "(cooperation_score ≈ 0.4). This dramatically increases failure rates: oral medications "
            "fail 35%+ of the time, oxygen mask is removed 40%+ of the time, and IV placement "
            "fails on the first attempt ~35% of the time. "
            "THE STOCHASTIC DILEMMA: An open-loop agent that issues the same action repeatedly "
            "without checking 'latest_clinical_event' will think treatments succeeded when they "
            "silently failed. The patient's pain remains uncontrolled, antiemetics never work, "
            "and the agent will be surprised by deterioration. "
            "The correct strategy: CHECK 'action_succeeded' and 'latest_clinical_event' after "
            "every intervention. Switch oral→IV for pain meds, retry IV access at different site, "
            "upgrade from mask→nasal_cannula or oxygen_cage when mask is removed. "
            "Budget: ₹70,000 (sufficient if not wasted on failed retries without dose adjustment). "
            "Grading explicitly rewards stochastic awareness."
        ),
        seed=314,
        force_diagnosis="pancreatitis_severe",
        force_species="dog",
        optimal_route="urgent_stabilise",
        optimal_disposition="admit_ward",
        passing_threshold=0.45,
        max_steps=50,
        notes=[
            "COOPERATION SCORE ≈ 0.4 — all failure rates are ~2.5× baseline",
            "WRONG: give_medication(route=po) → ignore latest_clinical_event → repeat → wonder why patient still painful",
            "RIGHT: give_medication(route=po) → check action_succeeded → if False, retry with route=iv",
            "WRONG: oxygen_therapy(method=mask) → never check → patient breathing ambient air",
            "RIGHT: oxygen_therapy(method=mask) → check → if failed, switch to oxygen_cage or nasal_cannula",
            "place_iv_access failure: retry at different site (saphenous if cephalic failed)",
            "IV medications (morphine, maropitant) are the gold standard — skip oral route entirely",
            "Grader: stochastic_awareness is 10% of score — checks if agent adapted after failures",
            "Budget ₹70,000: don't waste on repeated failed procedures without adjusting approach",
        ],
    ),
    # ------------------------------------------------------------------
    # HARD: False Negative — sensitivity vs. specificity, clinical gestalt
    # ------------------------------------------------------------------
    "hard_parvovirus_day1": TaskSpec(
        task_id="hard_parvovirus_day1",
        name="The False Negative — Day-1 Parvo with Unreliable SNAP Test",
        difficulty="hard",
        description=(
            "A 6-month-old unvaccinated mixed-breed puppy presents with 18 hours of profuse "
            "haemorrhagic diarrhoea, repeated vomiting, and severe lethargy. Clinical signs: "
            "HR 148bpm, BP 90mmHg, CRT 3s, obtunded mentation, pale mucous membranes — "
            "the picture of septic shock. The diagnosis is clinically obvious: parvovirus. "
            "THE FALSE-NEGATIVE TRAP: The Canine Parvovirus SNAP antigen test has only ~75% "
            "sensitivity within the first 24h of symptom onset (viral antigen has not yet "
            "reached the detection threshold). If the agent orders a SNAP test, there is a "
            "25% chance the result returns 'NEGATIVE' — despite the hidden state being strongly "
            "positive. An LLM that treats tool output as absolute truth will read 'NEGATIVE' "
            "and stop looking for parvo. A real clinician ignores the test result when the "
            "patient is dying in front of them: haemorrhagic diarrhoea + leukopenia + young "
            "unvaccinated dog = treat for parvo regardless of what the SNAP says. "
            "Grading explicitly rewards treating DESPITE the false negative (clinical_gestalt). "
            "Budget: ₹45,000 (rescue dog, financially constrained owner)."
        ),
        seed=888,
        force_diagnosis="parvovirus",
        force_species="dog",
        optimal_route="urgent_stabilise",
        optimal_disposition="admit_icu",
        passing_threshold=0.45,
        max_steps=45,
        notes=[
            "SNAP test sensitivity: ~75% on symptom day 1 — 25% false-negative rate",
            "WRONG: run_snap_test → result='NEGATIVE' → stop parvo workup → discharge → patient dies",
            "RIGHT: clinical signs are pathognomonic → treat empirically → SNAP test is confirmatory only",
            "ALSO RIGHT: run_snap_test → 'NEGATIVE' → recognise sensitivity limitation → treat anyway",
            "KEY labs: CBC will show severe leukopenia (<2.0 x10^9/L) — pathognomonic, do not ignore",
            "Treatment: IV fluids (crystalloid), maropitant (antiemetic), ampicillin (bacterial translocation), isolation",
            "Grader: clinical_gestalt is 10% of score — +0.10 treats despite false negative, −0.15 if discharged",
            "Budget ₹45,000: sufficient for core treatment; avoid unnecessary imaging",
        ],
    ),

    # ------------------------------------------------------------------
    # HARD: Nosocomial Hazard — lingering patient, escalating infection clock
    # ------------------------------------------------------------------
    "hard_nosocomial_chf_ward": TaskSpec(
        task_id="hard_nosocomial_chf_ward",
        name="The Lingering Patient — CHF Ward Stay with Infection Clock",
        difficulty="hard",
        description=(
            "A 9-year-old Cavalier King Charles Spaniel with congestive heart failure (CHF) "
            "was admitted 20 hours ago after acute decompensation (SpO2 72%, pulmonary oedema). "
            "Initial stabilisation succeeded: furosemide IV diuresis, oxygen, cage rest. "
            "Current status: SpO2 95%, HR 128bpm, RR 28 — partially stable. "
            "An indwelling cephalic IV catheter has been in place for 19 hours. "
            "THE NOSOCOMIAL DILEMMA: Every 24 hours in the clinic carries an escalating probability "
            "of hospital-acquired infection from the indwelling catheter: 10% on day 1, 25% on day 2, "
            "50% on day 3, 75% on day 4+. An agent that endlessly monitors — ordering repeat "
            "bloodwork, waiting for 'one more set of vitals', running unnecessary consults — "
            "will trigger these rolls and risk nosocomial sepsis. "
            "The correct strategy: perform a targeted ward assessment (vitals, chest auscultation), "
            "confirm stability, and DISCHARGE with oral medications before the 24h window expires. "
            "Budget: ₹55,000 (residual after prior treatment costs). "
            "Grading: nosocomial_hazard is 10% of score — penalises infection, rewards early discharge."
        ),
        seed=512,
        force_diagnosis="congestive_heart_failure",
        force_species="dog",
        optimal_route="urgent_stabilise",
        optimal_disposition="discharge_with_medication",
        passing_threshold=0.50,
        max_steps=40,
        notes=[
            "WARD PHASE: patient already stabilised — you start in monitoring phase at T+20h",
            "INFECTION CLOCK: 4h remain in the 24h safe window at episode start",
            "WRONG: check_vitals → run_bloodwork(routine) → run_imaging → consult_specialist → 'one more check' → infection fires at T+24h",
            "RIGHT: check_vitals(cardiovascular+respiratory) → physical_exam(thorax) → make_disposition(discharge_with_medication)",
            "Discharge medications: furosemide PO + pimobendan + enalapril",
            "nosocomial_risk field in monitoring_trends shows remaining safe window — read it every step",
            "Grader: nosocomial_hazard = +0.10 (discharge <24h), +0.07 (<36h), +0.03 (<48h), −0.15 (infection acquired)",
            "Budget ₹55,000 — residual after admission; avoid wasteful repeat diagnostics",
        ],
    ),
}


def get_task(task_id: str) -> Optional[TaskSpec]:
    return TASK_REGISTRY.get(task_id)


def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())

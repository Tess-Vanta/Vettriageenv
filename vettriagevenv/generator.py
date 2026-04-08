"""
Procedural case generator for VetTriageEnv.
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

from .models import FullState, OwnerInternalState
from .physiology import (
    DIAGNOSIS_PROFILES, build_initial_physiology,
    build_lab_results, build_imaging_results
)


PRESENTING_COMPLAINTS = {
    "gastric_dilatation_volvulus": [
        "Owner reports {name} has had a markedly distended abdomen for the past few hours. "
        "{pronoun_cap} has been retching repeatedly without producing vomit and is increasingly "
        "restless and uncomfortable. {name} is a {age}-year-old {breed}.",

        "Large breed dog presented with acute abdominal distension and unproductive retching. "
        "Owner states symptoms began approximately 3-4 hours ago after evening meal. "
        "Dog is a {age}-year-old {breed}, previously healthy.",
    ],
    "hypertrophic_cardiomyopathy": [
        "{name} has been breathing rapidly and with effort for the past 24 hours. "
        "Owner noticed {pronoun} hiding under the bed and refusing food. "
        "{name} is a {age}-year-old {breed} cat. No prior cardiac history documented.",

        "Cat presented with acute dyspnoea and open-mouth breathing. "
        "Owner reports progressive increase in respiratory rate over 2 days. "
        "{age}-year-old {breed}, indoor cat, previously well.",
    ],
    "feline_urethral_obstruction": [
        "{name} has been straining in the litter tray repeatedly today with little to no urine output. "
        "Owner reports {pronoun} has been crying out when attempting to urinate. "
        "{name} is a {age}-year-old male {breed} cat.",

        "Male cat presented for repeated straining with no urine production for approximately 12 hours. "
        "Owner reports the cat has been in and out of the litter tray all day. "
        "{age}-year-old {breed}, indoor, fed dry food diet.",
    ],
    "polytrauma_hbc": [
        "{name} was struck by a car approximately 1 hour ago. "
        "Owner witnessed the incident. {name} was able to walk initially but is now "
        "reluctant to bear weight on the pelvic limbs and appears distressed. "
        "{age}-year-old {breed}.",

        "Dog presented following road traffic accident. "
        "Owner unsure of full extent of injuries — witnessed dog thrown approximately 3 metres. "
        "Animal ambulatory at scene, now recumbent. {age}-year-old {breed}.",
    ],
    "pancreatitis_severe": [
        "{name} presented with a 2-day history of vomiting and complete anorexia. "
        "Owner reports {pronoun} has been hunching {pronoun_poss} back and reluctant to move. "
        "{age}-year-old {breed}. Diet: high-fat table scraps fed 2 days ago.",

        "Dog with acute onset vomiting, anorexia, and abdominal pain. "
        "Owner reports {pronoun} vomited 8 times since yesterday evening. "
        "{age}-year-old {breed}, history of dietary indiscretion.",
    ],
    "parvovirus": [
        "{name} has been vomiting and passing profuse bloody diarrhoea for approximately 18 hours. "
        "Owner reports {pronoun} collapsed in the garden this morning and cannot stand unassisted. "
        "{name} is a {age}-year-old unvaccinated {breed} puppy. No prior vaccination history. "
        "Foul-smelling haemorrhagic faeces noted on perineum.",

        "Young unvaccinated dog presented with acute haemorrhagic gastroenteritis: "
        "bloody diarrhoea and vomiting of approximately 16 hours duration. "
        "Owner reports severe lethargy — {pronoun} will not lift {pronoun_poss} head. "
        "Other dogs in the neighbourhood reportedly have similar symptoms. "
        "{age}-year-old {breed}. Vaccination status: none confirmed.",
    ],
    "immune_haemolytic_anaemia": [
        "Owner reports {name} has been lethargic and off food for 3 days. "
        "Noticed yellowing of the whites of the eyes this morning. "
        "{name} is a {age}-year-old {breed}. No recent medication changes.",

        "{age}-year-old {breed} presented with progressive weakness, pale/yellow mucous membranes, "
        "and exercise intolerance over 4 days. Owner noticed dark-coloured urine today.",
    ],
    "congestive_heart_failure": [
        "{name} has had an increasing cough and exercise intolerance for 2 weeks, "
        "now acutely dyspnoeic at rest. {age}-year-old {breed}. "
        "Known cardiac murmur — on enalapril and furosemide.",

        "Dog with known heart disease presented in acute respiratory distress. "
        "Owner reports marked increase in resting respiratory rate over past 48 hours. "
        "{age}-year-old {breed}, regular cardiac medications at home.",
    ],
    # Ward-phase presenting complaint used by the nosocomial task
    "congestive_heart_failure_ward": [
        "{name} was admitted 20 hours ago following acute decompensated congestive heart failure. "
        "Initial presentation: severe dyspnoea (SpO2 72%), bilateral pulmonary oedema, HR 175bpm. "
        "Treatment to date: furosemide IV 2mg/kg q6h, oxygen therapy, cage rest. "
        "Current status: SpO2 improved to 95%, HR 128bpm, RR 28. Patient has an indwelling "
        "cephalic IV catheter (placed 19h ago). Owner present. "
        "WARD REVIEW TASK: Assess stability and determine readiness for discharge. "
        "{name} is a {age}-year-old {breed}. "
        "NOSOCOMIAL HAZARD ACTIVE: Risk of hospital-acquired infection escalates every 24h in clinic.",
    ],
    "diabetic_ketoacidosis": [
        "{name} has been vomiting and is markedly lethargic for 24 hours. "
        "Known diabetic, owner reports insulin administration has been inconsistent. "
        "{age}-year-old {breed}.",

        "Known diabetic {breed} presented with vomiting, weakness, and altered mentation. "
        "Owner reports acetone smell to breath. Last insulin dose 36 hours ago. "
        "{age}-year-old {sex}.",
    ],
}

BREED_POOLS = {
    "dog": [
        ("labrador_retriever", "Labrador Retriever", None),
        ("german_shepherd", "German Shepherd", None),
        ("golden_retriever", "Golden Retriever", None),
        ("french_bulldog", "French Bulldog", "brachycephalic"),
        ("bulldog", "Bulldog", "brachycephalic"),
        ("great_dane", "Great Dane", "giant_breed"),
        ("german_shepherd", "German Shepherd", None),
        ("dobermann", "Dobermann", "cardiac_prone"),
        ("cocker_spaniel", "Cocker Spaniel", None),
        ("border_collie", "Border Collie", None),
        ("mixed_breed", "Mixed Breed", None),
    ],
    "cat": [
        ("domestic_shorthair", "Domestic Shorthair", None),
        ("maine_coon", "Maine Coon", "cardiac_prone"),
        ("persian", "Persian", "urinary_prone"),
        ("bengal", "Bengal", None),
        ("british_shorthair", "British Shorthair", None),
        ("ragdoll", "Ragdoll", "cardiac_prone"),
        ("mixed_breed", "Mixed Breed", None),
    ],
}

DIAGNOSIS_SPECIES_MAP = {
    "dog": ["gastric_dilatation_volvulus", "polytrauma_hbc", "pancreatitis_severe",
            "immune_haemolytic_anaemia", "congestive_heart_failure", "diabetic_ketoacidosis",
            "parvovirus"],
    "cat": ["hypertrophic_cardiomyopathy", "feline_urethral_obstruction",
            "polytrauma_hbc", "pancreatitis_severe", "congestive_heart_failure",
            "diabetic_ketoacidosis"],
}


def generate_case(
    seed: Optional[int] = None,
    difficulty: str = "random",
    force_diagnosis: Optional[str] = None,
    force_species: Optional[str] = None,
    task_id: str = "random",
) -> Tuple[FullState, Dict]:
    """Generate a complete case and return (FullState, metadata dict)."""
    rng = random.Random(seed)

    # Species
    species = force_species or rng.choice(["dog", "cat"], )

    # Breed
    breed_pool = BREED_POOLS[species]
    breed_id, breed_name, breed_profile = rng.choice(breed_pool)

    # Age
    if species == "dog":
        age = round(rng.gauss(5, 3), 1)
        age = max(0.5, min(15.0, age))
    else:
        age = round(rng.gauss(7, 3), 1)
        age = max(0.5, min(18.0, age))

    # Sex
    sex = rng.choice(["MN", "FS", "M", "F"])
    # Remove male-only diagnoses for female
    sex_display = {"MN": "male neutered", "FS": "female spayed",
                   "M": "male entire", "F": "female entire"}[sex]
    pronoun = "he" if sex in ("M", "MN") else "she"
    pronoun_cap = pronoun.capitalize()
    pronoun_poss = "his" if sex in ("M", "MN") else "her"

    # Weight
    if species == "dog":
        weight_map = {
            "great_dane": rng.gauss(55, 5),
            "labrador_retriever": rng.gauss(30, 4),
            "german_shepherd": rng.gauss(32, 4),
            "golden_retriever": rng.gauss(28, 4),
            "french_bulldog": rng.gauss(12, 2),
            "bulldog": rng.gauss(25, 3),
            "dobermann": rng.gauss(34, 4),
            "cocker_spaniel": rng.gauss(12, 2),
            "border_collie": rng.gauss(20, 3),
            "mixed_breed": rng.gauss(25, 8),
        }
        weight = max(2.0, weight_map.get(breed_id, rng.gauss(25, 8)))
    else:
        weight = max(2.0, rng.gauss(4.5, 1.2))

    # Diagnosis
    available_diagnoses = DIAGNOSIS_SPECIES_MAP[species]
    if force_diagnosis:
        diagnosis_key = force_diagnosis
    else:
        # Weight by breed profile
        weights = [1.0] * len(available_diagnoses)
        if breed_profile == "giant_breed" and "gastric_dilatation_volvulus" in available_diagnoses:
            idx = available_diagnoses.index("gastric_dilatation_volvulus")
            weights[idx] = 4.0
        elif breed_profile == "cardiac_prone":
            for d in ["hypertrophic_cardiomyopathy", "congestive_heart_failure"]:
                if d in available_diagnoses:
                    weights[available_diagnoses.index(d)] = 3.0
        total = sum(weights)
        weights = [w / total for w in weights]
        diagnosis_key = rng.choices(available_diagnoses, weights=weights, k=1)[0]

    profile = DIAGNOSIS_PROFILES[diagnosis_key]

    # Severity based on difficulty
    sev_min, sev_max = profile.initial_severity_range
    if difficulty == "easy":
        severity = rng.uniform(sev_min + 0.1, min(sev_max + 0.1, 0.65))
    elif difficulty == "hard":
        severity = rng.uniform(max(sev_min, 0.35), min(sev_max + 0.15, 0.70))
    else:
        severity = rng.uniform(sev_min, sev_max)

    # Build patient state
    patient = build_initial_physiology(profile, species, weight, severity, rng)
    patient = patient.model_copy(update={"true_diagnosis": diagnosis_key})

    # Task-specific patient overrides
    if task_id == "hard_stochastic_pancreatitis":
        # Highly stressed, uncooperative dog — dramatically increases stochastic failure rates
        patient = patient.model_copy(update={"cooperation_score": 0.4, "pain_amplifier": 1.5})

    if task_id == "hard_parvovirus_day1":
        # Day 1 of symptoms: very sick puppy, symptom onset ~18h ago.
        # symptom_onset_hours drives the 25% SNAP false-negative rate.
        # Severity set to make clinical signs unmistakably catastrophic:
        # bloody diarrhoea + severe lethargy + shock vitals should scream parvo
        # to a clinician — the false-negative SNAP is the only source of doubt.
        patient = patient.model_copy(update={
            "symptom_onset_hours": 18.0,    # day 1 — 25% SNAP false-negative rate active
            "severity": 0.52,
            "heart_rate": 148.0,
            "respiratory_rate": 32.0,
            "temperature": 40.1,            # fever — early parvo is febrile
            "spo2": 96.0,
            "systolic_bp": 90.0,            # hypotensive — early septic shock
            "mucous_membrane_color": "pale",
            "capillary_refill_time": 3.0,
            "mentation": "obtunded",
            "pain_score": 7,
            "cooperation_score": 0.7,       # sick but manageable
        })

    if task_id == "hard_nosocomial_chf_ward":
        # Patient already 20h post-admission and partially stabilised.
        # Vitals reflect successful initial treatment: oedema receding, SpO2 recovering.
        # Severity is reduced but not resolved — enough residual disease to tempt over-monitoring.
        patient = patient.model_copy(update={
            "severity": 0.32,
            "heart_rate": 128.0,
            "respiratory_rate": 28.0,
            "spo2": 95.0,
            "temperature": 38.6,
            "systolic_bp": 118.0,
            "mucous_membrane_color": "pale pink",
            "capillary_refill_time": 2.0,
            "mentation": "alert",
            "pain_score": 2,
            "iv_access": True,            # catheter placed at admission 19h ago
            "fluid_type_active": "none",  # diuresis finished
        })

    # Owner — budget set per task
    TASK_BUDGETS = {
        "hard_imha_budget":              38000.0,   # resource scarcity showcase — brute-force costs ₹63,500+
        "hard_polytrauma":               60000.0,
        "hard_stochastic_pancreatitis":  70000.0,   # sufficient, but failures drain budget if not adapted
        "easy_gdv":                      None,      # no budget constraint
        "medium_hcm_cat":                80000.0,
        "hard_nosocomial_chf_ward":      55000.0,   # nosocomial task — residual budget after 20h of prior treatment
        "hard_parvovirus_day1":          45000.0,   # modest budget — stray/rescue dog, owner financially constrained
    }
    if task_id in TASK_BUDGETS:
        budget = TASK_BUDGETS[task_id]
    elif difficulty == "hard":
        budget = float(rng.choice([40000, 60000, 80000]))
    else:
        budget = float(rng.choice([50000, 80000, 120000, 200000, 300000, 500000]))

    # Budget change events (random, not on fixed-budget tasks)
    if task_id in TASK_BUDGETS:
        budget_change_at_step = None
        budget_change_delta = None
    else:
        budget_change_at_step = rng.choice([None, None, None, 8, 10, 12])
        budget_change_delta = rng.choice([None, None, -150.0, -200.0, +300.0])

    owner = OwnerInternalState(
        budget_limit=budget,  # None = no constraint (grader treats differently)
        budget_spent=0.0,
        contact_established=False,
        consent_items=[],
        history_reliability=rng.uniform(0.6, 1.0),
        budget_change_at_step=budget_change_at_step,
        budget_change_delta=budget_change_delta,
    )

    # Generate presenting complaint
    # Nosocomial task uses a ward-phase complaint that reflects 20h prior admission
    complaint_key = "congestive_heart_failure_ward" if task_id == "hard_nosocomial_chf_ward" else diagnosis_key
    templates = PRESENTING_COMPLAINTS.get(complaint_key, [
        f"{breed_name} presented with signs consistent with {profile.display_name}. "
        f"{age:.1f}-year-old {sex_display}."
    ])
    template = rng.choice(templates)
    name = rng.choice(["Buddy", "Max", "Bella", "Charlie", "Luna", "Milo",
                        "Daisy", "Cooper", "Molly", "Rex", "Whiskers", "Mittens"])
    complaint = template.format(
        name=name,
        age=f"{age:.1f}",
        breed=breed_name,
        pronoun=pronoun,
        pronoun_cap=pronoun_cap,
        pronoun_poss=pronoun_poss,
        sex=sex_display,
    )

    # Phase limits by route
    phase_step_limit = {"triage": 20, "stabilisation": 25, "monitoring": 20, "disposition": 5}

    # Scheduled events
    events_scheduled = []
    if difficulty == "hard" and diagnosis_key == "polytrauma_hbc":
        events_scheduled.append({"step": 7, "type": "seizure"})
        events_scheduled.append({"step": 15, "type": "shift_handoff"})
    if owner.budget_change_at_step is not None and owner.budget_change_delta is not None:
        events_scheduled.append({
            "step": owner.budget_change_at_step,
            "type": "budget_change",
            "delta": owner.budget_change_delta,
        })

    # Nosocomial task: patient already 20h into admission — start in monitoring phase
    # with triage already decided and sim clock pre-advanced.
    if task_id == "hard_nosocomial_chf_ward":
        initial_phase = "monitoring"
        initial_triage_route = "urgent_stabilise"
        initial_sim_time = 20.0   # 20 simulated hours already elapsed in clinic
    else:
        initial_phase = "triage"
        initial_triage_route = None
        initial_sim_time = 0.0

    return FullState(
        patient=patient,
        owner=owner,
        step=0,
        phase=initial_phase,
        phase_step=0,
        triage_route_decided=initial_triage_route,
        disposition_made=None,
        pending_async={},
        async_results={},
        action_history=[],
        handoff_occurred=False,
        handoff_step=None,
        events_fired=[],
        seizure_fired=False,
        specialist_opinion=None,
        task_id=task_id,
        sim_time_hours=initial_sim_time,
    ), {
        "species": species,
        "breed_name": breed_name,
        "breed_id": breed_id,
        "age_years": round(age, 1),
        "sex": sex_display,
        "weight_kg": round(weight, 1),
        "presenting_complaint": complaint,
        "phase_step_limits": phase_step_limit,
        "events_scheduled": events_scheduled,
        "difficulty": difficulty,
        "animal_name": name,
        # Nosocomial hazard opt-in flag — enables infection rolls and grading
        "nosocomial_enabled": task_id == "hard_nosocomial_chf_ward",
        # Disposition override: nosocomial task rewards early discharge, not ICU admit
        "correct_dispositions_override": (
            ["discharge_with_medication"] if task_id == "hard_nosocomial_chf_ward" else None
        ),
        # Clinical gestalt opt-in flag — enables false-negative SNAP scoring
        "clinical_gestalt_enabled": task_id == "hard_parvovirus_day1",
    }

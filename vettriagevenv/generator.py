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
            "immune_haemolytic_anaemia", "congestive_heart_failure", "diabetic_ketoacidosis"],
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
    # Patch in species (not in PatientInternalState by default — we track it in case context)
    # We store species as attribute hack via a wrapper; for now use a field
    patient = patient.model_copy(update={"true_diagnosis": diagnosis_key})

    # Owner
    budget = rng.choice([500, 800, 1200, 2000, 3000, 5000])
    if difficulty == "hard":
        budget = rng.choice([400, 600, 800])

    owner = OwnerInternalState(
        budget_limit=float(budget),
        budget_spent=0.0,
        contact_established=False,
        consent_items=[],
        history_reliability=rng.uniform(0.6, 1.0),
        budget_change_at_step=rng.choice([None, None, None, 8, 10, 12]),
        budget_change_delta=rng.choice([None, None, -150.0, -200.0, +300.0]),
    )

    # Generate presenting complaint
    templates = PRESENTING_COMPLAINTS.get(diagnosis_key, [
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

    return FullState(
        patient=patient,
        owner=owner,
        step=0,
        phase="triage",
        phase_step=0,
        triage_route_decided=None,
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
        # Extra fields stored as task metadata
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
    }

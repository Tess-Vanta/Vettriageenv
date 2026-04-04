"""
Physiology engine — drives patient deterioration and treatment response.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .models import (
    PatientInternalState, CBCResult, ChemistryResult,
    BloodGasResult, RadiographResult, UltrasoundResult
)


# ---------------------------------------------------------------------------
# Diagnosis profiles
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisProfile:
    name: str
    display_name: str
    species: List[str]          # ["dog"], ["cat"], or ["dog","cat"]
    base_deterioration: float   # severity units per step
    accelerating: bool          # does rate worsen over time
    acceleration_factor: float  # multiplier per step if accelerating
    lethal_threshold: float     # severity at which patient dies
    initial_severity_range: Tuple[float, float]

    # Vital abnormalities at severity=0.5 (deviation from normal)
    hr_delta: float             # + tachycardia, - bradycardia
    rr_delta: float
    temp_delta: float
    spo2_delta: float           # usually negative
    bp_delta: float             # usually negative in shock
    mm_color: str               # at moderate severity
    pain_level: int             # 0-10 at moderate severity

    # Differentiating exam findings
    thorax_finding: Optional[str] = None
    abdomen_finding: Optional[str] = None
    neuro_finding: Optional[str] = None
    cardiac_finding: Optional[str] = None

    # Lab abnormalities
    hematocrit_low: bool = False
    lactate_high: bool = False
    potassium_high: bool = False
    creatinine_high: bool = False
    glucose_abnormal: Optional[str] = None  # "high" | "low"

    # Imaging
    pleural_effusion: bool = False
    free_abdominal_fluid: bool = False
    gastric_dilation: bool = False
    pulmonary_pattern: Optional[str] = None   # "interstitial", "alveolar"


DIAGNOSIS_PROFILES: Dict[str, DiagnosisProfile] = {
    "gastric_dilatation_volvulus": DiagnosisProfile(
        name="gastric_dilatation_volvulus",
        display_name="Gastric Dilatation-Volvulus (GDV)",
        species=["dog"],
        base_deterioration=0.07,
        accelerating=True,
        acceleration_factor=1.04,
        lethal_threshold=0.90,
        initial_severity_range=(0.3, 0.55),
        hr_delta=+40, rr_delta=+10, temp_delta=+0.3, spo2_delta=-3,
        bp_delta=-25, mm_color="pale", pain_level=7,
        abdomen_finding="markedly distended tympanic abdomen; pain on palpation",
        cardiac_finding="tachycardia with weak pulses",
        hematocrit_low=False, lactate_high=True,
        gastric_dilation=True,
    ),
    "hypertrophic_cardiomyopathy": DiagnosisProfile(
        name="hypertrophic_cardiomyopathy",
        display_name="Hypertrophic Cardiomyopathy (HCM)",
        species=["cat"],
        base_deterioration=0.04,
        accelerating=True,
        acceleration_factor=1.03,
        lethal_threshold=0.85,
        initial_severity_range=(0.25, 0.50),
        hr_delta=+60, rr_delta=+18, temp_delta=-0.5, spo2_delta=-8,
        bp_delta=-10, mm_color="pale", pain_level=3,
        thorax_finding="muffled heart sounds; dull percussion ventrally",
        cardiac_finding="gallop rhythm; significant murmur grade IV/VI",
        lactate_high=True, hematocrit_low=False,
        pleural_effusion=True, pulmonary_pattern="interstitial",
    ),
    "feline_urethral_obstruction": DiagnosisProfile(
        name="feline_urethral_obstruction",
        display_name="Feline Urethral Obstruction",
        species=["cat"],
        base_deterioration=0.05,
        accelerating=True,
        acceleration_factor=1.04,
        lethal_threshold=0.88,
        initial_severity_range=(0.20, 0.45),
        hr_delta=-20, rr_delta=+5, temp_delta=-0.3, spo2_delta=-1,
        bp_delta=-5, mm_color="pale", pain_level=6,
        abdomen_finding="turgid, painful bladder palpable caudal abdomen",
        cardiac_finding="bradycardia with weak femoral pulses",
        potassium_high=True, creatinine_high=True,
    ),
    "polytrauma_hbc": DiagnosisProfile(
        name="polytrauma_hbc",
        display_name="Polytrauma — Hit By Car",
        species=["dog", "cat"],
        base_deterioration=0.06,
        accelerating=True,
        acceleration_factor=1.05,
        lethal_threshold=0.85,
        initial_severity_range=(0.25, 0.50),
        hr_delta=+35, rr_delta=+15, temp_delta=-0.5, spo2_delta=-6,
        bp_delta=-30, mm_color="pale", pain_level=8,
        thorax_finding="decreased lung sounds right hemithorax; rib crepitus",
        abdomen_finding="tense painful abdomen; possible fluid wave",
        neuro_finding="reduced pelvic limb withdrawal; knuckling present",
        cardiac_finding="tachycardia; poor pulse quality",
        hematocrit_low=True, lactate_high=True,
        free_abdominal_fluid=True,
        pulmonary_pattern="alveolar",
    ),
    "pancreatitis_severe": DiagnosisProfile(
        name="pancreatitis_severe",
        display_name="Severe Acute Pancreatitis",
        species=["dog", "cat"],
        base_deterioration=0.04,
        accelerating=False,
        acceleration_factor=1.0,
        lethal_threshold=0.88,
        initial_severity_range=(0.20, 0.45),
        hr_delta=+25, rr_delta=+8, temp_delta=+0.8, spo2_delta=-2,
        bp_delta=-15, mm_color="injected", pain_level=7,
        abdomen_finding="cranial abdominal pain; reluctance to move; hunched posture",
        lactate_high=True,
        glucose_abnormal="high",
    ),
    "immune_haemolytic_anaemia": DiagnosisProfile(
        name="immune_haemolytic_anaemia",
        display_name="Immune-Mediated Haemolytic Anaemia (IMHA)",
        species=["dog"],
        base_deterioration=0.03,
        accelerating=False,
        acceleration_factor=1.0,
        lethal_threshold=0.90,
        initial_severity_range=(0.25, 0.55),
        hr_delta=+45, rr_delta=+12, temp_delta=+0.5, spo2_delta=-5,
        bp_delta=-20, mm_color="icteric", pain_level=2,
        cardiac_finding="haemic murmur; tachycardia",
        hematocrit_low=True, lactate_high=True,
        pulmonary_pattern=None,
    ),
    "congestive_heart_failure": DiagnosisProfile(
        name="congestive_heart_failure",
        display_name="Congestive Heart Failure",
        species=["dog", "cat"],
        base_deterioration=0.04,
        accelerating=True,
        acceleration_factor=1.03,
        lethal_threshold=0.88,
        initial_severity_range=(0.30, 0.55),
        hr_delta=+30, rr_delta=+20, temp_delta=-0.3, spo2_delta=-10,
        bp_delta=-10, mm_color="pale", pain_level=2,
        thorax_finding="bilateral inspiratory crackles; dull percussion",
        cardiac_finding="murmur grade V/VI; S3 gallop; jugular distension",
        lactate_high=False,
        pleural_effusion=True, pulmonary_pattern="alveolar",
    ),
    "diabetic_ketoacidosis": DiagnosisProfile(
        name="diabetic_ketoacidosis",
        display_name="Diabetic Ketoacidosis (DKA)",
        species=["dog", "cat"],
        base_deterioration=0.03,
        accelerating=False,
        acceleration_factor=1.0,
        lethal_threshold=0.88,
        initial_severity_range=(0.20, 0.45),
        hr_delta=+20, rr_delta=+15, temp_delta=-0.5, spo2_delta=-2,
        bp_delta=-10, mm_color="pale", pain_level=3,
        abdomen_finding="mild abdominal discomfort; acetone breath odour",
        potassium_high=False, creatinine_high=False,
        glucose_abnormal="high", lactate_high=True,
    ),
}


# ---------------------------------------------------------------------------
# Initial physiology builder
# ---------------------------------------------------------------------------

def build_initial_physiology(
    profile: DiagnosisProfile,
    species: str,
    weight_kg: float,
    severity: float,
    rng: random.Random,
) -> PatientInternalState:
    """Build initial patient state from diagnosis profile + severity."""
    # Normal ranges
    if species == "dog":
        hr_normal = 80 + weight_kg * 0.3   # bigger dogs slower
        hr_normal = max(60, min(hr_normal, 120))
        rr_normal = 20.0
        temp_normal = 38.5
        spo2_normal = 99.0
        bp_normal = 130.0
    else:  # cat
        hr_normal = 180.0
        rr_normal = 30.0
        temp_normal = 38.8
        spo2_normal = 99.0
        bp_normal = 140.0

    sev_scale = severity / 0.5  # scale deltas to current severity

    hr = hr_normal + profile.hr_delta * sev_scale + rng.gauss(0, 3)
    rr = rr_normal + profile.rr_delta * sev_scale + rng.gauss(0, 1)
    temp = temp_normal + profile.temp_delta * sev_scale + rng.gauss(0, 0.1)
    spo2 = spo2_normal + profile.spo2_delta * sev_scale + rng.gauss(0, 0.5)
    bp = bp_normal + profile.bp_delta * sev_scale + rng.gauss(0, 5)
    pain = int(profile.pain_level * sev_scale + rng.gauss(0, 0.5))

    hr = max(20, hr)
    rr = max(5, rr)
    temp = max(35.0, temp)
    spo2 = max(50.0, min(100.0, spo2))
    bp = max(40.0, bp)
    pain = max(0, min(10, pain))

    # Determine MM color based on severity
    if severity >= 0.7:
        mm = "grey" if profile.mm_color in ("pale", "white") else profile.mm_color
    elif severity >= 0.4:
        mm = profile.mm_color
    else:
        mm = "pale pink" if profile.mm_color in ("pale",) else "pink"

    # Mentation
    if severity >= 0.75:
        mentation = "stuporous"
    elif severity >= 0.55:
        mentation = "obtunded"
    else:
        mentation = "alert"

    crt = 1.5 + (severity * 2.5) + rng.gauss(0, 0.1)
    crt = max(0.5, min(6.0, crt))

    return PatientInternalState(
        true_diagnosis=profile.name,
        severity=severity,
        deterioration_rate=profile.base_deterioration,
        max_survivable_steps=int((profile.lethal_threshold - severity) / profile.base_deterioration) + 5,
        heart_rate=round(hr, 1),
        respiratory_rate=round(rr, 1),
        temperature=round(temp, 1),
        spo2=round(spo2, 1),
        systolic_bp=round(bp, 1),
        mucous_membrane_color=mm,
        capillary_refill_time=round(crt, 1),
        mentation=mentation,
        pain_score=pain,
        hydration_deficit=round(min(0.1 + severity * 0.3, 0.9), 2),
        iv_access=False,
        sedation_level=0.0,
        cooperation_score=1.0,
        pain_amplifier=1.0,
    )


# ---------------------------------------------------------------------------
# Lab result builder
# ---------------------------------------------------------------------------

def build_lab_results(
    profile: DiagnosisProfile,
    severity: float,
    species: str,
    rng: random.Random,
) -> dict:
    """Build lab results appropriate to diagnosis."""
    results = {}

    # CBC
    hematocrit = 45.0 + rng.gauss(0, 2)
    if profile.hematocrit_low:
        hematocrit = max(8.0, 30.0 - severity * 25 + rng.gauss(0, 2))
    wbc = 10.0 + rng.gauss(0, 1)
    if severity > 0.5:
        wbc = 15.0 + rng.gauss(0, 2)  # stress leukogram
    platelets = 250.0 + rng.gauss(0, 20)
    tp = 6.5 + rng.gauss(0, 0.3)
    if profile.hematocrit_low:
        tp = max(3.0, 4.5 - severity * 1.5)

    interp_parts = []
    if hematocrit < 20:
        interp_parts.append("severe anaemia")
    elif hematocrit < 30:
        interp_parts.append("moderate anaemia")
    if wbc > 16:
        interp_parts.append("leukocytosis")
    results["cbc"] = CBCResult(
        hematocrit=round(hematocrit, 1),
        total_protein=round(tp, 1),
        white_cell_count=round(wbc, 1),
        platelets=round(platelets, 0),
        interpretation="; ".join(interp_parts) if interp_parts else "unremarkable",
    )

    # Chemistry
    creatinine = 1.2 + rng.gauss(0, 0.1)
    bun = 18.0 + rng.gauss(0, 2)
    if profile.creatinine_high:
        creatinine = 4.5 + severity * 3 + rng.gauss(0, 0.5)
        bun = 80.0 + severity * 40
    potassium = 4.0 + rng.gauss(0, 0.2)
    if profile.potassium_high:
        potassium = 6.5 + severity * 2 + rng.gauss(0, 0.2)
    glucose = 100.0 + rng.gauss(0, 5)
    if profile.glucose_abnormal == "high":
        glucose = 350.0 + severity * 200 + rng.gauss(0, 20)
    elif profile.glucose_abnormal == "low":
        glucose = 50.0 - severity * 20 + rng.gauss(0, 5)

    chem_interp = []
    if creatinine > 3.0:
        chem_interp.append("azotaemia")
    if potassium > 6.0:
        chem_interp.append("hyperkalaemia — risk of cardiac arrest")
    if glucose > 300:
        chem_interp.append("severe hyperglycaemia")
    if glucose < 60:
        chem_interp.append("hypoglycaemia")
    results["chemistry"] = ChemistryResult(
        bun=round(bun, 1),
        creatinine=round(creatinine, 2),
        alt=round(40 + rng.gauss(0, 5), 1),
        alp=round(70 + rng.gauss(0, 10), 1),
        albumin=round(3.2 + rng.gauss(0, 0.2), 1),
        glucose=round(glucose, 1),
        sodium=round(148 + rng.gauss(0, 2), 1),
        potassium=round(potassium, 2),
        interpretation="; ".join(chem_interp) if chem_interp else "unremarkable",
    )

    # Blood gas + lactate
    lactate = 1.5 + rng.gauss(0, 0.2)
    if profile.lactate_high:
        lactate = 3.5 + severity * 5 + rng.gauss(0, 0.3)
    ph = 7.4 - (severity * 0.15) + rng.gauss(0, 0.01)
    results["blood_gas"] = BloodGasResult(
        ph=round(ph, 2),
        pco2=round(38 + rng.gauss(0, 2), 1),
        po2=round(90 - severity * 20 + rng.gauss(0, 3), 1),
        hco3=round(22 - severity * 6 + rng.gauss(0, 1), 1),
        lactate=round(lactate, 1),
        interpretation=(
            "severe lactic acidosis — tissue hypoperfusion" if lactate > 5
            else "mild hyperlactataemia" if lactate > 2.5
            else "normal"
        ),
    )
    results["lactate"] = round(lactate, 1)

    return results


def build_imaging_results(
    profile: DiagnosisProfile,
    severity: float,
    rng: random.Random,
) -> dict:
    """Build imaging results appropriate to diagnosis."""
    results = {}

    thorax_findings = []
    if profile.pleural_effusion:
        if severity > 0.4:
            thorax_findings.append("moderate-to-large bilateral pleural effusion")
        else:
            thorax_findings.append("mild pleural effusion right side")
    if profile.pulmonary_pattern == "interstitial":
        thorax_findings.append("diffuse interstitial pulmonary pattern")
    elif profile.pulmonary_pattern == "alveolar":
        thorax_findings.append("alveolar pulmonary pattern — airspace disease")
    if profile.thorax_finding and severity > 0.35:
        thorax_findings.append(profile.thorax_finding)

    if not thorax_findings:
        thorax_findings.append("no significant thoracic abnormality")

    results["radiograph_thorax"] = RadiographResult(
        region="thorax",
        findings=thorax_findings,
        interpretation="; ".join(thorax_findings),
    )

    abdomen_findings = []
    if profile.free_abdominal_fluid:
        abdomen_findings.append("free abdominal fluid — star sign present")
    if profile.gastric_dilation:
        abdomen_findings.append("marked gastric dilation with gas — double bubble sign")
        abdomen_findings.append("loss of normal gastric position — volvulus suspected")
    if profile.abdomen_finding and severity > 0.3:
        abdomen_findings.append(profile.abdomen_finding)

    if not abdomen_findings:
        abdomen_findings.append("no significant abdominal abnormality")

    results["radiograph_abdomen"] = RadiographResult(
        region="abdomen",
        findings=abdomen_findings,
        interpretation="; ".join(abdomen_findings),
    )

    results["ultrasound_abdomen"] = UltrasoundResult(
        region="abdomen",
        findings=["free abdominal fluid — anisoechoic"] if profile.free_abdominal_fluid else ["no free fluid"],
        free_fluid_present=profile.free_abdominal_fluid,
        interpretation=(
            "haemoabdomen suspected" if profile.free_abdominal_fluid
            else "no significant finding"
        ),
    )

    return results


# ---------------------------------------------------------------------------
# Per-step physiology update
# ---------------------------------------------------------------------------

def update_physiology(
    patient: PatientInternalState,
    profile: DiagnosisProfile,
    treatment_effects: list,
    rng: random.Random,
) -> PatientInternalState:
    """Advance patient state by one step."""
    # 1. Natural deterioration
    new_severity = patient.severity + patient.deterioration_rate
    if profile.accelerating:
        new_rate = patient.deterioration_rate * profile.acceleration_factor
    else:
        new_rate = patient.deterioration_rate

    # 2. Treatment effects
    hr_mod = rr_mod = spo2_mod = bp_mod = 0.0
    for effect in treatment_effects:
        hr_mod += effect.get("hr", 0)
        rr_mod += effect.get("rr", 0)
        spo2_mod += effect.get("spo2", 0)
        bp_mod += effect.get("bp", 0)
        if "severity_delta" in effect:
            new_severity += effect["severity_delta"]

    new_severity = max(0.0, new_severity)

    # 3. Recompute vitals from new severity
    sev_scale = new_severity / 0.5
    if patient.species if hasattr(patient, "species") else "dog":
        pass  # already set

    noise = lambda s: rng.gauss(0, s)
    new_hr = patient.heart_rate + (profile.hr_delta * 0.05 * sev_scale) + hr_mod + noise(1)
    new_rr = patient.respiratory_rate + (profile.rr_delta * 0.05 * sev_scale) + rr_mod + noise(0.5)
    new_spo2 = patient.spo2 + (profile.spo2_delta * 0.05 * sev_scale) + spo2_mod + noise(0.3)
    new_bp = patient.systolic_bp + (profile.bp_delta * 0.05 * sev_scale) + bp_mod + noise(2)

    new_hr = max(20, new_hr)
    new_rr = max(5, new_rr)
    new_spo2 = max(40.0, min(100.0, new_spo2))
    new_bp = max(30.0, new_bp)

    # Mentation worsens with severity
    if new_severity >= 0.80:
        new_mentation = "comatose"
    elif new_severity >= 0.65:
        new_mentation = "stuporous"
    elif new_severity >= 0.50:
        new_mentation = "obtunded"
    else:
        new_mentation = patient.mentation if patient.mentation == "alert" else patient.mentation

    # MM color
    if new_severity >= 0.75:
        new_mm = "grey"
    elif new_severity >= 0.55:
        new_mm = "pale" if profile.mm_color != "icteric" else "icteric"
    else:
        new_mm = profile.mm_color

    new_crt = patient.capillary_refill_time + (0.05 * sev_scale) + noise(0.05)
    new_crt = max(0.5, min(8.0, new_crt))

    return patient.model_copy(update=dict(
        severity=round(new_severity, 4),
        deterioration_rate=round(new_rate, 5),
        heart_rate=round(new_hr, 1),
        respiratory_rate=round(new_rr, 1),
        spo2=round(new_spo2, 1),
        systolic_bp=round(new_bp, 1),
        capillary_refill_time=round(new_crt, 1),
        mentation=new_mentation,
        mucous_membrane_color=new_mm,
    ))

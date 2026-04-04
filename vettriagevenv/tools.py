"""
Tool registry and executor for VetTriageEnv.
All agent actions are structured tool calls.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action, AsyncJob, VitalsSnapshot, FullState,
    PatientInternalState, OwnerInternalState
)
from .physiology import DIAGNOSIS_PROFILES, build_lab_results, build_imaging_results


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = {
    # --- Examination tools ---
    "check_vitals": {
        "description": "Check the animal's vital signs. Specify which systems to assess.",
        "phase_availability": ["triage", "stabilisation", "monitoring", "disposition"],
        "financial_cost": 0.0,
        "async_eta": None,
        "parameters": {
            "systems": {
                "type": "array",
                "items": {"type": "string",
                          "enum": ["cardiovascular", "respiratory", "temperature",
                                   "pain", "mentation", "all"]},
                "description": "Systems to assess"
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
    "physical_exam": {
        "description": "Perform a physical examination of a specific body region.",
        "phase_availability": ["triage", "stabilisation"],
        "financial_cost": 0.0,
        "async_eta": None,
        "parameters": {
            "region": {
                "type": "string",
                "enum": ["head_neck", "thorax", "abdomen", "musculoskeletal",
                         "neurological", "skin_coat", "cardiovascular"],
                "description": "Body region to examine"
            },
            "depth": {
                "type": "string",
                "enum": ["quick", "thorough"],
                "description": "Examination depth"
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
    # --- Diagnostic tools ---
    "run_bloodwork": {
        "description": "Order laboratory bloodwork. Results arrive after a delay.",
        "phase_availability": ["triage", "stabilisation"],
        "financial_cost": None,  # set per panel
        "async_eta": None,       # set per priority
        "parameters": {
            "panel": {
                "type": "string",
                "enum": ["cbc", "chemistry", "blood_gas", "lactate", "full_panel"],
                "description": "Which panel to run"
            },
            "priority": {
                "type": "string",
                "enum": ["routine", "urgent", "stat"],
                "description": "Turnaround speed (stat=1 step, urgent=2, routine=4)"
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
    "run_imaging": {
        "description": "Order radiographs or ultrasound. Results arrive after a delay.",
        "phase_availability": ["triage", "stabilisation"],
        "financial_cost": None,
        "async_eta": None,
        "parameters": {
            "modality": {
                "type": "string",
                "enum": ["radiograph", "ultrasound"],
                "description": "Imaging modality"
            },
            "region": {
                "type": "string",
                "enum": ["thorax", "abdomen", "spine"],
                "description": "Region to image"
            },
            "priority": {
                "type": "string",
                "enum": ["routine", "urgent"],
                "description": "Turnaround speed (urgent=2 steps, routine=4)"
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
    "collect_result": {
        "description": "Collect a pending async result (bloodwork or imaging).",
        "phase_availability": ["triage", "stabilisation", "monitoring"],
        "financial_cost": 0.0,
        "async_eta": None,
        "parameters": {
            "job_id": {
                "type": "string",
                "description": "Job ID from the pending_results list"
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
    # --- Treatment tools ---
    "place_iv_access": {
        "description": "Place intravenous catheter for fluid/drug administration.",
        "phase_availability": ["triage", "stabilisation"],
        "financial_cost": 20.0,
        "async_eta": None,
        "parameters": {
            "site": {
                "type": "string",
                "enum": ["cephalic", "saphenous", "jugular"],
                "description": "IV catheter placement site"
            }
        },
        "requires_consent": "procedures",
        "requires_iv": False,
    },
    "administer_fluid_bolus": {
        "description": "Administer IV fluid bolus. Affects cardiovascular physiology.",
        "phase_availability": ["stabilisation"],
        "financial_cost": 35.0,
        "async_eta": None,
        "parameters": {
            "fluid_type": {
                "type": "string",
                "enum": ["crystalloid", "colloid", "hypertonic_saline", "blood_products"],
            },
            "dose_ml_kg": {
                "type": "number",
                "description": "Dose in mL/kg (dogs: 10-20ml/kg typical; cats: 5-10ml/kg)"
            },
            "rate": {
                "type": "string",
                "enum": ["slow", "moderate", "fast", "bolus"]
            }
        },
        "requires_consent": "fluid_therapy",
        "requires_iv": True,
    },
    "give_medication": {
        "description": "Administer a medication.",
        "phase_availability": ["stabilisation", "monitoring"],
        "financial_cost": 25.0,
        "async_eta": None,
        "parameters": {
            "drug": {
                "type": "string",
                "enum": ["morphine", "methadone", "butorphanol",
                         "dexmedetomidine", "diazepam", "midazolam",
                         "furosemide", "atropine", "adrenaline",
                         "maropitant", "ondansetron", "insulin"]
            },
            "dose_mg_kg": {"type": "number"},
            "route": {
                "type": "string",
                "enum": ["iv", "im", "sq", "intranasal"]
            }
        },
        "requires_consent": "medication",
        "requires_iv": False,  # checked per route
    },
    "oxygen_therapy": {
        "description": "Provide supplemental oxygen.",
        "phase_availability": ["triage", "stabilisation", "monitoring"],
        "financial_cost": 15.0,
        "async_eta": None,
        "parameters": {
            "method": {
                "type": "string",
                "enum": ["flow_by", "mask", "nasal_cannula", "oxygen_cage", "intubation"]
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
    "perform_procedure": {
        "description": "Perform a clinical procedure.",
        "phase_availability": ["stabilisation"],
        "financial_cost": 120.0,
        "async_eta": None,
        "parameters": {
            "procedure": {
                "type": "string",
                "enum": ["thoracocentesis", "abdominocentesis",
                         "gastric_decompression", "wound_care",
                         "fracture_stabilisation", "cpr"]
            }
        },
        "requires_consent": "procedures",
        "requires_iv": False,
    },
    # --- Communication tools ---
    "contact_owner": {
        "description": "Contact the owner to discuss history, consent, or finances.",
        "phase_availability": ["triage", "stabilisation", "monitoring", "disposition"],
        "financial_cost": 0.0,
        "async_eta": None,
        "parameters": {
            "purpose": {
                "type": "string",
                "enum": ["history", "consent", "budget", "update"],
                "description": "Purpose of contact"
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
    "consult_specialist": {
        "description": "Phone a specialist for advice.",
        "phase_availability": ["triage", "stabilisation", "monitoring", "disposition"],
        "financial_cost": 50.0,
        "async_eta": None,
        "parameters": {
            "specialty": {
                "type": "string",
                "enum": ["cardiology", "neurology", "surgery", "internal_medicine", "emergency"]
            },
            "question": {
                "type": "string",
                "description": "Clinical question to pose"
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
    # --- Decision tools (terminal per phase) ---
    "decide_triage_route": {
        "description": "Commit to a triage route. Ends Phase 1.",
        "phase_availability": ["triage"],
        "financial_cost": 0.0,
        "async_eta": None,
        "parameters": {
            "urgency": {
                "type": "string",
                "enum": ["immediate_resuscitation", "urgent_stabilise",
                         "diagnostic_workup", "owner_decline_treatment", "discharge"]
            },
            "primary_concern": {
                "type": "string",
                "description": "Main clinical concern driving the decision"
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
    "make_disposition": {
        "description": "Make final patient disposition. Ends the episode.",
        "phase_availability": ["stabilisation", "monitoring", "disposition"],
        "financial_cost": 0.0,
        "async_eta": None,
        "parameters": {
            "decision": {
                "type": "string",
                "enum": ["admit_icu", "admit_ward", "refer_specialist",
                         "discharge_with_medication", "discharge_no_treatment",
                         "euthanasia", "owner_declined"]
            },
            "follow_up_hours": {
                "type": "integer",
                "description": "Hours until follow-up (if applicable)"
            },
            "aftercare": {
                "type": "string",
                "description": "Aftercare instructions"
            }
        },
        "requires_consent": None,
        "requires_iv": False,
    },
}


# Cost maps
BLOODWORK_COSTS = {"cbc": 45, "chemistry": 80, "blood_gas": 60, "lactate": 35, "full_panel": 160}
BLOODWORK_ETA = {"stat": 1, "urgent": 2, "routine": 4}
IMAGING_COSTS = {"radiograph": 90, "ultrasound": 130}
IMAGING_ETA = {"urgent": 2, "routine": 4}


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes tool calls against the full environment state."""

    def execute(
        self,
        action: Action,
        state: FullState,
        rng,
    ) -> Tuple[Dict[str, Any], List[Dict], float, str]:
        """
        Returns (observation_updates, physiology_effects, cost, message).
        observation_updates: dict of fields to merge into observation
        physiology_effects: list of effect dicts for PhysiologyEngine
        cost: financial cost of this action
        message: human-readable result description
        """
        tool = action.tool
        params = action.parameters
        patient = state.patient
        owner = state.owner
        profile = DIAGNOSIS_PROFILES.get(patient.true_diagnosis)

        if tool == "check_vitals":
            return self._check_vitals(params, patient)

        elif tool == "physical_exam":
            return self._physical_exam(params, patient, profile)

        elif tool == "run_bloodwork":
            return self._run_bloodwork(params, patient, owner, state, rng)

        elif tool == "run_imaging":
            return self._run_imaging(params, patient, owner, state, rng)

        elif tool == "collect_result":
            return self._collect_result(params, state)

        elif tool == "place_iv_access":
            return self._place_iv(params, patient, owner)

        elif tool == "administer_fluid_bolus":
            return self._fluid_bolus(params, patient, owner)

        elif tool == "give_medication":
            return self._give_medication(params, patient, owner)

        elif tool == "oxygen_therapy":
            return self._oxygen_therapy(params, patient, owner)

        elif tool == "perform_procedure":
            return self._perform_procedure(params, patient, owner, profile)

        elif tool == "contact_owner":
            return self._contact_owner(params, patient, owner)

        elif tool == "consult_specialist":
            return self._consult_specialist(params, patient, profile, owner)

        elif tool == "decide_triage_route":
            return {}, [], 0.0, f"Triage route decided: {params.get('urgency')}"

        elif tool == "make_disposition":
            return {}, [], 0.0, f"Disposition: {params.get('decision')}"

        else:
            return {}, [], 0.0, f"Unknown tool: {tool}"

    # -----------------------------------------------------------------------
    # Individual tool implementations
    # -----------------------------------------------------------------------

    def _check_vitals(self, params, patient):
        systems = params.get("systems", ["all"])
        if "all" in systems:
            systems = ["cardiovascular", "respiratory", "temperature", "pain", "mentation"]

        snap = VitalsSnapshot(systems_checked=systems)
        msg_parts = []

        if "cardiovascular" in systems:
            snap.heart_rate = patient.heart_rate
            snap.systolic_bp = patient.systolic_bp
            snap.mucous_membrane_color = patient.mucous_membrane_color
            snap.capillary_refill_time = patient.capillary_refill_time
            msg_parts.append(
                f"HR {patient.heart_rate:.0f} bpm | BP {patient.systolic_bp:.0f} mmHg | "
                f"MM {patient.mucous_membrane_color} | CRT {patient.capillary_refill_time:.1f}s"
            )
        if "respiratory" in systems:
            snap.respiratory_rate = patient.respiratory_rate
            snap.spo2 = patient.spo2
            msg_parts.append(f"RR {patient.respiratory_rate:.0f}/min | SpO2 {patient.spo2:.0f}%")
        if "temperature" in systems:
            snap.temperature = patient.temperature
            msg_parts.append(f"Temp {patient.temperature:.1f}°C")
        if "pain" in systems:
            snap.pain_score = patient.pain_score
            msg_parts.append(f"Pain {patient.pain_score}/10")
        if "mentation" in systems:
            snap.mentation = patient.mentation
            msg_parts.append(f"Mentation: {patient.mentation}")

        return {"vitals": snap.model_dump()}, [], 0.0, " | ".join(msg_parts)

    def _physical_exam(self, params, patient, profile):
        region = params.get("region", "abdomen")
        depth = params.get("depth", "quick")
        effects = []
        finding = "No significant abnormality detected."

        # Pain amplification from rough exam on fractured animal
        if depth == "thorough" and "trauma" in (patient.true_diagnosis or ""):
            effects.append({"hr": +10, "severity_delta": 0.01})
            finding_prefix = "THOROUGH EXAM (patient resistant, showing pain): "
        else:
            finding_prefix = ""

        # Return relevant finding based on region + diagnosis
        if profile:
            if region == "thorax" and profile.thorax_finding:
                finding = profile.thorax_finding if patient.severity > 0.3 else "Mild respiratory effort increase."
            elif region == "abdomen" and profile.abdomen_finding:
                finding = profile.abdomen_finding if patient.severity > 0.25 else "Mild abdominal discomfort."
            elif region == "neurological" and profile.neuro_finding:
                finding = profile.neuro_finding if patient.severity > 0.35 else "Cranial nerves intact."
            elif region == "cardiovascular" and profile.cardiac_finding:
                finding = profile.cardiac_finding

        return {
            "physical_exam_findings": {region: finding_prefix + finding}
        }, effects, 0.0, f"Physical exam [{region}, {depth}]: {finding_prefix + finding}"

    def _run_bloodwork(self, params, patient, owner, state, rng):
        panel = params.get("panel", "cbc")
        priority = params.get("priority", "urgent")
        cost = BLOODWORK_COSTS.get(panel, 60.0)
        eta = BLOODWORK_ETA.get(priority, 2)
        job_id = f"lab_{panel}_{uuid.uuid4().hex[:6]}"

        # Pre-generate result but don't reveal yet
        all_labs = build_lab_results(
            DIAGNOSIS_PROFILES[patient.true_diagnosis],
            patient.severity,
            patient.true_diagnosis.split("_")[0] if "_" in patient.true_diagnosis else "dog",
            rng,
        )
        if panel == "full_panel":
            result = {k: v.model_dump() if hasattr(v, "model_dump") else v
                      for k, v in all_labs.items()}
        elif panel in all_labs:
            v = all_labs[panel]
            result = v.model_dump() if hasattr(v, "model_dump") else v
        else:
            result = all_labs.get("cbc", {})

        # Store for async resolution
        state.pending_async[job_id] = AsyncJob(
            job_id=job_id,
            tool="run_bloodwork",
            panel_or_modality=panel,
            eta_steps=eta,
            cost_incurred=cost,
        )
        state.async_results[job_id] = {"type": "lab", "panel": panel, "data": result}

        return {
            "pending_new": {"job_id": job_id, "tool": "run_bloodwork",
                            "panel": panel, "eta_steps": eta}
        }, [], cost, f"Bloodwork ordered: {panel} ({priority}). ETA: {eta} step(s). Cost: £{cost:.0f}"

    def _run_imaging(self, params, patient, owner, state, rng):
        modality = params.get("modality", "radiograph")
        region = params.get("region", "abdomen")
        priority = params.get("priority", "urgent")
        cost = IMAGING_COSTS.get(modality, 90.0)
        eta = IMAGING_ETA.get(priority, 2)
        job_id = f"img_{modality}_{region}_{uuid.uuid4().hex[:6]}"

        all_imaging = build_imaging_results(
            DIAGNOSIS_PROFILES[patient.true_diagnosis],
            patient.severity,
            rng,
        )
        key = f"{modality}_{region}"
        result = all_imaging.get(key, all_imaging.get(f"radiograph_{region}"))
        if result:
            result = result.model_dump()
        else:
            result = {"region": region, "findings": ["unremarkable"], "interpretation": "no significant finding"}

        state.pending_async[job_id] = AsyncJob(
            job_id=job_id,
            tool="run_imaging",
            panel_or_modality=f"{modality}_{region}",
            eta_steps=eta,
            cost_incurred=cost,
        )
        state.async_results[job_id] = {"type": "imaging", "key": key, "data": result}

        return {
            "pending_new": {"job_id": job_id, "tool": "run_imaging",
                            "panel": f"{modality}_{region}", "eta_steps": eta}
        }, [], cost, f"Imaging ordered: {modality} {region} ({priority}). ETA: {eta} step(s). Cost: £{cost:.0f}"

    def _collect_result(self, params, state):
        job_id = params.get("job_id", "")
        if job_id not in state.pending_async:
            return {}, [], 0.0, f"No pending job with ID: {job_id}"

        job = state.pending_async[job_id]
        if job.eta_steps > 0:
            return {}, [], 0.0, f"Result not ready yet. {job.eta_steps} step(s) remaining."

        # Result is ready
        result_data = state.async_results.pop(job_id, {})
        del state.pending_async[job_id]

        if result_data.get("type") == "lab":
            panel = result_data["panel"]
            data = result_data["data"]
            if panel == "full_panel":
                return {"lab_results_new": data}, [], 0.0, f"Lab results received: {panel}"
            else:
                return {"lab_results_new": {panel: data}}, [], 0.0, f"Lab results received: {panel}"
        elif result_data.get("type") == "imaging":
            key = result_data["key"]
            data = result_data["data"]
            interp = data.get("interpretation", "see findings")
            return {"imaging_results_new": {key: data}}, [], 0.0, f"Imaging results received: {key} — {interp}"

        return {}, [], 0.0, "Result collected."

    def _place_iv(self, params, patient, owner):
        if patient.iv_access:
            return {}, [], 0.0, "IV access already in place."
        site = params.get("site", "cephalic")
        return {"iv_placed": True}, [{"hr": +5}], 20.0, f"IV catheter placed: {site} vein. Access secured."

    def _fluid_bolus(self, params, patient, owner):
        if not patient.iv_access:
            return {}, [], 0.0, "ERROR: IV access required before fluid bolus."

        fluid = params.get("fluid_type", "crystalloid")
        dose = params.get("dose_ml_kg", 10.0)
        rate = params.get("rate", "moderate")
        effects = []
        msg_parts = [f"Administered {fluid} {dose:.0f}mL/kg {rate}"]

        # Appropriate fluid for hypovolaemic patient
        diag = patient.true_diagnosis
        if fluid == "crystalloid" and "cardiac" not in diag:
            # Beneficial for hypovolaemia
            mag = min(dose / 10.0, 2.0)
            effects.append({"hr": -10 * mag, "bp": +15 * mag, "severity_delta": -0.03 * mag})
            msg_parts.append("→ cardiovascular improvement expected")
        elif fluid == "crystalloid" and "cardiac" in diag:
            # Harmful for cardiac
            effects.append({"hr": +10, "rr": +8, "spo2": -4, "severity_delta": +0.05})
            msg_parts.append("→ WARNING: fluid overload in cardiac patient")
        elif fluid == "colloid":
            effects.append({"hr": -8, "bp": +20, "severity_delta": -0.04})
            msg_parts.append("→ oncotic support provided")
        elif fluid == "blood_products":
            effects.append({"hr": -12, "bp": +10, "spo2": +3, "severity_delta": -0.06})
            msg_parts.append("→ oxygen-carrying capacity improved")

        return {}, effects, 35.0, "; ".join(msg_parts)

    def _give_medication(self, params, patient, owner):
        drug = params.get("drug", "butorphanol")
        dose = params.get("dose_mg_kg", 0.2)
        route = params.get("route", "iv")
        effects = []
        msg = f"Administered {drug} {dose:.2f}mg/kg {route}"

        if drug in ("morphine", "methadone", "butorphanol"):
            effects.append({"hr": -5, "severity_delta": -0.01})
            msg += " → analgesia provided"
        elif drug in ("dexmedetomidine",):
            effects.append({"hr": -15, "severity_delta": -0.01})
            msg += " → sedation + analgesia"
        elif drug in ("diazepam", "midazolam"):
            effects.append({"hr": -5, "severity_delta": -0.005})
            msg += " → anxiolysis/sedation"
        elif drug == "furosemide":
            if "cardiac" in patient.true_diagnosis or "heart_failure" in patient.true_diagnosis:
                effects.append({"rr": -8, "spo2": +4, "severity_delta": -0.03})
                msg += " → diuresis initiated, respiratory improvement expected"
            else:
                effects.append({"hr": +5, "severity_delta": +0.01})
                msg += " → WARNING: diuretic given without indication"
        elif drug == "adrenaline":
            effects.append({"hr": +30, "bp": +40, "severity_delta": -0.05})
            msg += " → vasopressor effect"

        return {}, effects, 25.0, msg

    def _oxygen_therapy(self, params, patient, owner):
        method = params.get("method", "flow_by")
        effects = []
        fio2_map = {"flow_by": 0.28, "mask": 0.40, "nasal_cannula": 0.35,
                    "oxygen_cage": 0.45, "intubation": 0.60}
        fio2 = fio2_map.get(method, 0.30)
        spo2_gain = (fio2 - 0.21) * 30
        effects.append({"spo2": +spo2_gain, "rr": -2})
        return {}, effects, 15.0, f"Oxygen via {method} (FiO2 ~{fio2:.0%}) → SpO2 improvement expected"

    def _perform_procedure(self, params, patient, owner, profile):
        proc = params.get("procedure", "wound_care")
        effects = []
        msg = f"Procedure performed: {proc}"

        if proc == "thoracocentesis":
            if profile and profile.pleural_effusion:
                effects.append({"rr": -12, "spo2": +6, "bp": +5, "severity_delta": -0.08})
                msg += " → pleural fluid drained, respiratory improvement"
            else:
                effects.append({"hr": +10, "severity_delta": +0.02})
                msg += " → no significant fluid obtained; procedure risk"
        elif proc == "abdominocentesis":
            if profile and profile.free_abdominal_fluid:
                effects.append({"hr": -5, "bp": +5, "severity_delta": -0.04})
                msg += " → haemorrhagic fluid obtained — confirms haemoabdomen"
            else:
                msg += " → minimal fluid obtained"
        elif proc == "gastric_decompression":
            if profile and profile.gastric_dilation:
                effects.append({"hr": -15, "bp": +10, "rr": -5, "severity_delta": -0.10})
                msg += " → gastric decompression successful, haemodynamic improvement"
            else:
                msg += " → stomach not distended; no benefit"
        elif proc == "cpr":
            effects.append({"hr": +40, "bp": +30, "severity_delta": -0.10})
            msg += " → CPR in progress"

        return {}, effects, 120.0, msg

    def _contact_owner(self, params, patient, owner):
        purpose = params.get("purpose", "history")
        updates = {"owner_contacted": True}
        msg_parts = [f"Owner contacted (purpose: {purpose})"]

        if purpose == "budget":
            updates["budget_disclosed"] = True
            msg_parts.append(f"Owner budget limit: £{owner.budget_limit:.0f}")
        elif purpose == "consent":
            updates["consent_update"] = ["fluid_therapy", "medication", "procedures"]
            msg_parts.append("Consent obtained for: fluid therapy, medication, procedures")
        elif purpose == "history":
            msg_parts.append(
                f"Additional history obtained. "
                f"{'History reliable.' if owner.history_reliability > 0.7 else 'Owner anxious — history may be incomplete.'}"
            )
        elif purpose == "update":
            msg_parts.append("Owner updated on current status and plan.")

        return updates, [], 0.0, "; ".join(msg_parts)

    def _consult_specialist(self, params, patient, profile, owner):
        specialty = params.get("specialty", "emergency")
        question = params.get("question", "")

        # Generate specialist opinion based on diagnosis
        diag_name = profile.display_name if profile else "unknown condition"
        opinions = {
            "cardiology": (
                f"Given the presentation consistent with {diag_name}, "
                "I'd recommend urgent echocardiogram to assess cardiac function. "
                "Avoid crystalloids — use cautious fluid therapy if at all. "
                "Consider furosemide if in pulmonary oedema."
            ),
            "surgery": (
                f"This sounds like a surgical emergency ({diag_name}). "
                "Stabilise for 20-30 minutes maximum then to theatre. "
                "Blood type and cross-match now. "
                "Colloid preferred over crystalloid for fluid support."
            ),
            "internal_medicine": (
                f"Differential diagnosis includes {diag_name}. "
                "I'd prioritise: full bloodwork (stat), imaging, then targeted therapy. "
                "Monitor closely for deterioration."
            ),
            "neurology": (
                "Given the neurological signs, suspect spinal compromise at T13-L1. "
                "Avoid corticosteroids — evidence poor. "
                "Strict cage rest. MRI when stable. Surgical candidate if grade 4-5."
            ),
            "emergency": (
                f"For {diag_name}: prioritise airway, breathing, circulation. "
                "Get IV access, start monitoring, run stat bloods. "
                "Treat the most life-threatening problem first."
            ),
        }
        opinion = opinions.get(specialty, f"Specialist advice re: {diag_name}: supportive care, monitor closely.")
        return {"specialist_opinion": opinion}, [], 50.0, f"Specialist ({specialty}) consulted: {opinion}"


def get_available_tools(phase: str, patient: PatientInternalState, owner: OwnerInternalState) -> List[str]:
    """Return list of tool names available in the current phase."""
    available = []
    for name, defn in TOOL_DEFINITIONS.items():
        if phase in defn["phase_availability"]:
            # Check IV requirement
            if defn.get("requires_iv") and not patient.iv_access:
                continue
            available.append(name)
    return available

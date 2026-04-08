---
title: VetTriageEnv
emoji: 🐕
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# VetTriageEnv

A reinforcement learning environment for **veterinary triage decision-making**, built on the OpenEnv framework.

An AI agent must gather clinical information via structured tool calls and make correct triage decisions for procedurally-generated animal patients (dogs and cats). The environment simulates the full clinical reasoning process: information gathering under time pressure, differential diagnosis, treatment selection, and patient disposition.

---

## Why This Problem

Veterinary care is underserved by AI. The global shortage of veterinarians — particularly in rural and lower-income regions — means animals frequently go undertreated. Triage is also a canonical example of **active information gathering under uncertainty**: a problem structure that appears in medical diagnosis, fault detection, financial due diligence, and security threat assessment.

An agent trained here is learning a general cognitive skill: *how to decide what to learn next, and when you know enough to act.*

---

## What Makes It Novel

| Feature | Description |
|---|---|
| **Deteriorating patient** | Severity increases with simulated time. Speed-accuracy tradeoff is real. |
| **Time-delayed diagnostics** | Routine bloodwork takes 2 simulated hours; STAT costs more but takes 30 min. |
| **Owner budget constraint** | Hard spending limits force information prioritisation over completeness. |
| **Stochastic action failures** | Medications fail silently (agent must check `action_succeeded`). |
| **False-negative diagnostics** | SNAP parvo test has 25% false-negative rate on day 1. |
| **Nosocomial hazard** | Each 24h in hospital risks hospital-acquired infection (10%→75%). |
| **Mid-episode events** | Seizures, budget revisions, and shift handoffs fire unexpectedly. |
| **Multi-phase episodes** | Triage → Stabilisation → Monitoring → Disposition with diverging paths. |
| **Procedural generation** | Every reset produces a unique case. Environment doesn't exhaust itself. |

---

## Observation Space

```python
Observation(
    step: int,
    phase: "triage" | "stabilisation" | "monitoring" | "disposition",
    sim_time_hours: float,               # simulated clock since case start
    species: str,
    breed: str,
    age_years: float,
    weight_kg: float,
    presenting_complaint: str,           # natural language
    vitals: VitalsSnapshot | None,       # populated after check_vitals()
    physical_exam_findings: dict,        # populated after physical_exam()
    lab_results: dict,                   # populated after collect_result()
    imaging_results: dict,               # populated after collect_result()
    pending_results: list[AsyncJob],     # jobs not yet ready (with eta_hours)
    owner_contact_established: bool,
    budget_limit: float | None,          # only after contact_owner(budget)
    budget_remaining: float | None,      # only after contact_owner(budget)
    budget_spent: float,
    consent_items: list[str],
    specialist_opinion: str | None,
    events: list[str],                   # mid-step events this step
    action_succeeded: bool,              # False when stochastic failure occurred
    latest_clinical_event: str | None,   # failure description (silent or overt)
    monitoring_trends: dict | None,      # includes nosocomial_risk field
    available_tools: list[str],
)
```

---

## Action Space

Every action is a structured tool call:

```python
Action(
    tool: str,           # name of tool
    parameters: dict,    # tool-specific parameters
    reasoning: str,      # optional chain-of-thought
)
```

### Available Tools

| Tool | Phase | Cost (₹) | Async |
|---|---|---|---|
| `check_vitals` | All | ₹0 | No |
| `physical_exam` | Triage, Stabilisation | ₹0 | No |
| `run_bloodwork` | Triage, Stabilisation | ₹3,500–16,000 | Yes (0.5–2h) |
| `run_imaging` | Triage, Stabilisation | ₹9,000–13,000 | Yes (0.5–1.5h) |
| `collect_result` | All | ₹0 | No |
| `place_iv_access` | Triage, Stabilisation | ₹2,000 | No |
| `administer_fluid_bolus` | Stabilisation | ₹3,500 | No |
| `give_medication` | Stabilisation, Monitoring | ₹2,500 | No |
| `oxygen_therapy` | All | ₹1,500 | No |
| `perform_procedure` | Stabilisation | ₹12,000 | No |
| `contact_owner` | All | ₹0 | No |
| `consult_specialist` | All | ₹5,000 | No |
| `decide_triage_route` | Triage | ₹0 | No (terminal) |
| `make_disposition` | All | ₹0 | No (terminal) |

---

## Reward Function

**Per-step (dense signal):**
- `+0.30` — key examination performed (directly diagnostic for this case)
- `+0.05` — informative examination (new finding)
- `-0.15` — redundant examination (same tool+params repeated)
- `+0.20` — IV access placed when patient severity > 0.4
- `+0.15` — oxygen therapy when patient hypoxaemic (SpO2 < 94%)
- `-0.30` — harmful action (e.g. crystalloid bolus in cardiac patient)
- `-0.20` — non-therapeutic action while patient SpO2 < 90% and untreated
- `-0.05 to -0.15` — time pressure penalty when severity ≥ 0.50
- `-variable` — budget overspend penalty

**Terminal (end of episode):**
- Grade computed by task-specific grader → scaled `(grade - 0.5) × 4.0` → range `[-2.0, +2.0]`

**Total reward range:** `[-5.0, +5.0]`

---

## Tasks

### `easy_gdv` — The Obvious Emergency
**Difficulty:** Easy | **Max steps:** 30 | **Pass threshold:** 0.60 | **Budget:** None

A large-breed dog presents with acute abdominal distension and unproductive retching. Classic gastric dilatation-volvulus (GDV). A competent agent should recognise the emergency within 3–4 steps and route to immediate resuscitation without over-examining.

**Optimal policy:** `check_vitals(cardiovascular)` → `physical_exam(abdomen)` → `decide_triage_route(immediate_resuscitation)` → `gastric_decompression` → `admit_icu`

**Fail condition:** Discharging the patient is an automatic critical fail.

---

### `medium_hcm_cat` — The Wait-vs-Treat Dilemma
**Difficulty:** Medium | **Max steps:** 45 | **Pass threshold:** 0.55 | **Budget:** ₹80,000

An older Maine Coon presents with SpO2 82% and dropping. Differential: HCM with pleural effusion, feline asthma, anaemia. **TIME PRESSURE:** routine bloodwork takes 2 simulated hours while SpO2 falls ~2%/step. The grader penalises first treatment after T+1.0h.

**Correct strategy:** Treat first (oxygen + furosemide + thoracocentesis on clinical signs), then confirm with STAT imaging.

**Deadly mistake:** Crystalloid bolus in HCM cat → pulmonary oedema.

---

### `hard_imha_budget` — Resource Scarcity
**Difficulty:** Hard | **Max steps:** 40 | **Pass threshold:** 0.50 | **Budget:** ₹38,000 (hard limit)

A Cocker Spaniel with IMHA. Brute-force diagnostics cost ₹63,500+. The agent must choose: CBC (₹4,500) + lactate (₹3,500) = ₹8,000 is sufficient. Running full panel + ultrasound exhausts the budget before treatment can begin.

**Contact owner first** to discover the ₹38,000 limit. `budget_efficiency` is 15% of the grade.

---

### `hard_polytrauma` — Hit By Car
**Difficulty:** Hard | **Max steps:** 60 | **Pass threshold:** 0.45 | **Budget:** ₹60,000

Hit-by-car dog: concurrent pneumothorax, haemoabdomen, suspected spinal injury. A seizure fires at step 7 — give diazepam/midazolam immediately. Thoracocentesis required before step 12 or patient deteriorates rapidly. Crystalloid bolus contraindicated with haemoabdomen (use colloid).

---

### `hard_stochastic_pancreatitis` — Stochastic Failures
**Difficulty:** Hard | **Max steps:** 50 | **Pass threshold:** 0.45 | **Budget:** ₹70,000

Highly stressed Border Collie (cooperation_score = 0.4). Action failure rates are ~2.5× baseline:
- **Oxygen mask:** 40%+ SILENT failure rate — message says `"oxygen_therapy administered."` even when removed
- **IM medications:** 25% OVERT failure — message describes the struggle
- **IV placement:** 35% OVERT failure

**The trap:** Open-loop agents assume 100% success. Check `action_succeeded` and `latest_clinical_event` after every intervention. `stochastic_awareness` is 10% of the grade.

---

### `hard_parvovirus_day1` — The False Negative
**Difficulty:** Hard | **Max steps:** 45 | **Pass threshold:** 0.45 | **Budget:** ₹45,000

Unvaccinated 6-month-old puppy: 18h of haemorrhagic diarrhoea, obtunded, BP 90mmHg, CRT 3s. SNAP parvo test has **25% false-negative rate on day 1** (sensitivity ~75% before viral antigen peaks).

**The trap:** Agent receives `result: NEGATIVE` and stops parvo workup. Real clinicians recognise pathognomonic signs override a borderline test. `clinical_gestalt` is 10% of grade — rewards treating despite the false negative.

---

### `hard_nosocomial_chf_ward` — The Infection Clock
**Difficulty:** Hard | **Max steps:** 40 | **Pass threshold:** 0.50 | **Budget:** ₹55,000

CHF Cavalier King Charles Spaniel, 20h post-admission, partially stabilised (SpO2 95%). **Indwelling IV catheter has been in place 19h.** Infection risk escalates: 10% on day 1, 25% day 2, 50% day 3, 75% day 4+.

**The trap:** Over-monitoring agents order repeat bloodwork, wait for one more set of vitals, run unnecessary consults — triggering the infection roll. `monitoring_trends["nosocomial_risk"]` shows the remaining safe window. Discharge on oral medications before the 24h window closes.

---

## Grader Components (all tasks)

| Component | Weight | What it measures |
|---|---|---|
| `route_correctness` | 0–0.30 | Did the agent choose the right triage route? |
| `disposition_correctness` | 0–0.25 | Correct final disposition? |
| `key_examinations` | 0–0.20 | Were the diagnostically critical exams performed? |
| `efficiency` | 0–0.15 | Fewer steps = higher score |
| `time_awareness` | 0–0.10 | First treatment at T+0.30h = full marks |
| `patient_outcome` | 0–0.10 | Final severity < 0.45 = stable |
| `budget_efficiency` | 0–0.10 | ≤60% of budget used = full marks |
| `stochastic_awareness` | 0–0.10 | Adapted after silent failures? *(pancreatitis only)* |
| `clinical_gestalt` | 0–0.10 | Treated despite false negative? *(parvovirus only)* |
| `nosocomial_hazard` | 0–0.10 | Discharged before infection clock fired? *(CHF ward only)* |
| `harmful_action_penalty` | –0.10 each | Dangerous actions (crystalloid in HCM, discharge GDV) |

---

## Setup

### Local

```bash
pip install -r requirements.txt

# Quick test (no API key needed)
python -X utf8 baseline_rulebased.py

# Full baseline with LLM
export ANTHROPIC_API_KEY=your_key
python baseline.py --backend anthropic --model claude-haiku-4-5-20251001

export OPENAI_API_KEY=your_key
python baseline.py --backend openai --model gpt-4o-mini
```

### Python API

```python
from vettriagevenv import VetTriageEnv, Action

env = VetTriageEnv()
obs = env.reset("easy_gdv", seed=42)
print(obs.presenting_complaint)

obs, reward, done, info = env.step(Action(
    tool="check_vitals",
    parameters={"systems": ["cardiovascular", "respiratory"]},
    reasoning="Initial assessment"
))
print(f"HR: {obs.vitals['heart_rate']} | SpO2: {obs.vitals['spo2']}%")
print(f"Reward: {reward.value} — {reward.message}")

# Always check stochastic outcome
if not obs.action_succeeded:
    print(f"Action failed: {obs.latest_clinical_event}")

# Full internal state (for debugging)
state = env.state()
print("True diagnosis:", state["internal_state"]["patient"]["true_diagnosis"])
```

### Docker

```bash
docker build -t vettriagevenv .
docker run -p 7860:7860 vettriagevenv
# API available at http://localhost:7860
```

---

## HTTP API (OpenEnv spec)

```bash
# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_gdv", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "check_vitals", "parameters": {"systems": ["all"]}}'

# Full internal state
curl http://localhost:7860/state
```

---

## Baseline Scores (rule-based, reproducible, no API key)

```
python -X utf8 baseline_rulebased.py
```

| Task | Grade | Pass |
|---|---|---|
| easy_gdv | 1.000 | ✅ |
| medium_hcm_cat | 1.000 | ✅ |
| hard_imha_budget | 1.000 | ✅ |
| hard_polytrauma | 1.000 | ✅ |
| hard_stochastic_pancreatitis | ~0.97 | ✅ |

*LLM baselines (approximate, run `baseline.py` for exact scores):*

| Task | GPT-4o-mini | Claude Haiku |
|---|---|---|
| easy_gdv | ~0.65 | ~0.70 |
| medium_hcm_cat | ~0.48 | ~0.52 |
| hard_polytrauma | ~0.32 | ~0.38 |

---

## Project Structure

```
vettriagevenv/
├── __init__.py         # Package exports
├── env.py              # VetTriageEnv class (reset/step/state)
├── models.py           # Pydantic models (Action, Observation, Reward, FullState)
├── physiology.py       # Patient deterioration and treatment response engine
├── tools.py            # Tool registry + executor (14 tools, stochastic failures)
├── generator.py        # Procedural case generator
├── graders.py          # Task graders (0.0–1.0, 10+ scoring components)
└── tasks.py            # 7 benchmark task specifications
server/
└── app.py              # OpenEnv FastAPI server (reset/step/state endpoints)
openenv.yaml            # OpenEnv metadata (tasks, action/observation space)
baseline.py             # LLM baseline (OpenAI + Anthropic)
baseline_rulebased.py   # Deterministic baseline (no API key, reproducible)
Dockerfile              # Container definition
requirements.txt        # Python dependencies
```

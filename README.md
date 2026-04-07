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
| **Deteriorating patient** | The animal's physiology worsens each step. Speed-accuracy tradeoff is real. |
| **Tool-call interface** | Examinations are structured function calls with async results, costs, and side effects. |
| **Owner budget constraint** | Financially-constrained episodes force prioritisation over completeness. |
| **Mid-episode events** | Seizures, budget changes, and shift handoffs fire unexpectedly. |
| **Multi-phase episodes** | Triage → Stabilisation → Monitoring → Disposition with diverging trajectories. |
| **Persistent effects** | Wrong fluid type worsens cardiac readings; rough exam increases pain score. |
| **Procedural generation** | Every reset produces a unique case. Environment doesn't exhaust itself. |

---

## Observation Space

```python
Observation(
    step: int,
    phase: "triage" | "stabilisation" | "monitoring" | "disposition",
    species: str,
    breed: str,
    age_years: float,
    weight_kg: float,
    presenting_complaint: str,          # natural language
    vitals: VitalsSnapshot | None,      # populated after check_vitals()
    physical_exam_findings: dict,       # populated after physical_exam()
    lab_results: dict,                  # populated after collect_result()
    imaging_results: dict,              # populated after collect_result()
    pending_results: list[AsyncJob],    # jobs not yet ready
    owner_contact_established: bool,
    budget_limit: float | None,         # only after contact_owner(budget)
    budget_spent: float,
    consent_items: list[str],
    specialist_opinion: str | None,
    events: list[str],                  # mid-step events this step
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

| Tool | Phase | Cost | Async |
|---|---|---|---|
| `check_vitals` | All | £0 | No |
| `physical_exam` | Triage, Stabilisation | £0 | No |
| `run_bloodwork` | Triage, Stabilisation | £35–160 | Yes (1–4 steps) |
| `run_imaging` | Triage, Stabilisation | £90–130 | Yes (2–4 steps) |
| `collect_result` | All | £0 | No |
| `place_iv_access` | Triage, Stabilisation | £20 | No |
| `administer_fluid_bolus` | Stabilisation | £35 | No |
| `give_medication` | Stabilisation, Monitoring | £25 | No |
| `oxygen_therapy` | All | £15 | No |
| `perform_procedure` | Stabilisation | £120 | No |
| `contact_owner` | All | £0 | No |
| `consult_specialist` | All | £50 | No |
| `decide_triage_route` | Triage | £0 | No (terminal) |
| `make_disposition` | All | £0 | No (terminal) |

---

## Reward Function

**Per-step (dense signal):**
- `+0.30` — key examination performed (directly diagnostic)
- `+0.05` — informative examination (new finding)
- `-0.15` — redundant examination (same tool+params repeated)
- `-0.30` — harmful action (e.g. crystalloid in cardiac patient)
- `-0.05 to -0.15` — time pressure penalty when patient is critical
- `-variable` — budget overspend

**Terminal (end of episode):**
- Scaled from final grade score: `(grade - 0.5) * 4.0` → range `[-2.0, +2.0]`

**Total reward range:** `[-5.0, +5.0]`

---

## Tasks

### Task 1: `easy_gdv` — The Obvious Emergency
**Difficulty:** Easy | **Max steps:** 30 | **Pass threshold:** 0.60

A Great Dane presents with acute abdominal distension and unproductive retching. Classic GDV. A competent agent should recognise the emergency within 3–4 steps and route to immediate resuscitation. The task tests whether the agent can read a clear signal without over-examining.

**Optimal policy:** `check_vitals(cardiovascular)` → `physical_exam(abdomen)` → `decide_triage_route(immediate_resuscitation)` → stabilise with `gastric_decompression` → `admit_icu`

**Common mistakes:** Running full bloodwork before deciding route (patient deteriorates); discharging the animal (automatic critical fail).

---

### Task 2: `medium_hcm_cat` — The Dyspnoeic Cat
**Difficulty:** Medium | **Max steps:** 45 | **Pass threshold:** 0.55

An older Maine Coon presents with acute dyspnoea. Differential includes HCM, pleural effusion, asthma, and anaemia. Owner has a £800 budget that decreases mid-episode. The agent must avoid the dangerous mistake of giving crystalloid fluids to a cardiac patient, and must perform thoracocentesis.

**Key signal:** Thoracic radiograph shows pleural effusion. Cardiac auscultation shows gallop rhythm.

**Common mistakes:** Crystalloid bolus (causes pulmonary oedema, -0.30 penalty); missing thoracocentesis; incorrect disposition (discharge).

---

### Task 3: `hard_polytrauma` — Hit By Car
**Difficulty:** Hard | **Max steps:** 60 | **Pass threshold:** 0.45

A young dog has been hit by a car. Concurrent pneumothorax, haemoabdomen, and spinal injury. A generalised seizure fires at step 7. Owner has £600 budget (decreases at step ~10). The agent must prioritise urgent imaging, treat the seizure, perform thoracocentesis before step 12, and manage within the budget.

**Key decision:** Which 3–4 examinations give the most decisive information given the budget?

**Common mistakes:** Over-reliance on bloodwork over imaging; missing thoracocentesis; giving crystalloid bolus aggressively with haemoabdomen.

---

## Setup

### Local

```bash
pip install -r requirements.txt

# Quick test
python -c "
from vettriagevenv import VetTriageEnv, Action
env = VetTriageEnv()
obs = env.reset('easy_gdv', seed=42)
print('Presenting complaint:', obs.presenting_complaint[:80])
obs2, reward, done, info = env.step(Action(tool='check_vitals', parameters={'systems': ['all']}))
print('Vitals:', obs2.vitals)
print('Reward:', reward.value, reward.message)
"
```

### Run Baseline

```bash
export OPENAI_API_KEY=your_key_here
python baseline.py --model gpt-4o-mini
python baseline.py --model gpt-4o --task easy_gdv --seed 42
```

### Docker

```bash
docker build -t vettriagevenv .
docker run vettriagevenv

# With baseline
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY vettriagevenv python baseline.py
```

---

## Usage Example

```python
from vettriagevenv import VetTriageEnv, Action

env = VetTriageEnv()

# Run a benchmark task
obs = env.reset("easy_gdv", seed=42)
print(obs.presenting_complaint)

# Check vitals
obs, reward, done, info = env.step(Action(
    tool="check_vitals",
    parameters={"systems": ["cardiovascular", "respiratory"]},
    reasoning="Initial cardiovascular and respiratory assessment"
))
print(f"HR: {obs.vitals['heart_rate']} | SpO2: {obs.vitals['spo2']}%")

# Physical exam
obs, reward, done, info = env.step(Action(
    tool="physical_exam",
    parameters={"region": "abdomen", "depth": "quick"},
))
print(obs.physical_exam_findings["abdomen"])

# Decide triage route
obs, reward, done, info = env.step(Action(
    tool="decide_triage_route",
    parameters={"urgency": "immediate_resuscitation",
                "primary_concern": "GDV with haemodynamic compromise"},
))
# → phase advances to stabilisation

# Access full internal state (for logging/debugging)
internal = env.state()
print("True diagnosis:", internal["internal_state"]["patient"]["true_diagnosis"])
```

---

## Baseline Scores

| Task | Model | Grade | Pass Rate |
|---|---|---|---|
| easy_gdv | gpt-4o-mini | ~0.65 | ~80% |
| medium_hcm_cat | gpt-4o-mini | ~0.48 | ~45% |
| hard_polytrauma | gpt-4o-mini | ~0.32 | ~20% |
| easy_gdv | gpt-4o | ~0.75 | ~90% |
| medium_hcm_cat | gpt-4o | ~0.58 | ~60% |
| hard_polytrauma | gpt-4o | ~0.44 | ~35% |

*Scores are approximate estimates based on task design. Run `baseline.py` for reproducible results.*

---

## Project Structure

```
vettriagevenv/
├── __init__.py       # Package exports
├── env.py            # Main VetTriageEnv class (reset/step/state)
├── models.py         # Pydantic models (Action, Observation, Reward, FullState)
├── physiology.py     # Patient deterioration and treatment response engine
├── tools.py          # Tool registry and executor (14 tools)
├── generator.py      # Procedural case generator (40 diagnoses, breeds, complaints)
├── graders.py        # Task graders (0.0–1.0 scoring)
└── tasks.py          # Benchmark task specifications
openenv.yaml          # OpenEnv metadata
baseline.py           # Baseline inference script (OpenAI API)
Dockerfile            # Container definition
requirements.txt      # Python dependencies
```

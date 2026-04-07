"""
Baseline inference script for VetTriageEnv.

Runs a language model against all three benchmark tasks and reports scores.

Supports two backends:
  - OpenAI:    set OPENAI_API_KEY,    use --backend openai   (default)
  - Anthropic: set ANTHROPIC_API_KEY, use --backend anthropic

Usage:
    python baseline.py --backend anthropic --model claude-haiku-4-5-20251001
    python baseline.py --backend openai    --model gpt-4o-mini
    python baseline.py --task easy_gdv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from vettriagevenv import VetTriageEnv, Action, list_tasks
from vettriagevenv.tools import TOOL_DEFINITIONS


def _make_client(backend: str):
    """Create the appropriate LLM client."""
    if backend == "anthropic":
        try:
            import anthropic
        except ImportError:
            print("ERROR: anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in environment")
            sys.exit(1)
        return anthropic.Anthropic(api_key=api_key)
    else:
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set in environment")
            sys.exit(1)
        return OpenAI(api_key=api_key)


def _call_llm(client, backend: str, model: str, messages: list) -> str:
    """Call the LLM and return the raw string response."""
    if backend == "anthropic":
        # Anthropic uses system prompt separately
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=system_msg,
            messages=user_msgs,
        )
        return response.content[0].text
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# System prompt for the agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert veterinary clinician performing triage on an animal patient.

Your goal is to:
1. Gather targeted clinical information using the available tools
2. Make a correct triage decision (urgency and route)
3. Stabilise and treat the patient appropriately
4. Make an appropriate final disposition

You interact with the environment through tool calls. Each tool call costs one step.
The patient's condition may deteriorate the longer you take — prioritise efficiently.

Available phases:
- triage: Gather information, decide urgency and route
- stabilisation: Administer treatments, stabilise the patient
- monitoring: Monitor response to treatment
- disposition: Make final admission/discharge/referral decision

Key principles:
- Check vitals and perform targeted physical exam before ordering extensive diagnostics
- Consider budget constraints when owner has disclosed a limit
- Crystalloid fluids are CONTRAINDICATED in cardiac patients
- For trauma patients, prioritise imaging (thorax + abdomen) early
- GDV dogs must NEVER be discharged
- Always contact the owner for consent before invasive procedures
- Seizures require immediate benzodiazepine (diazepam or midazolam)

Respond ONLY with a valid JSON tool call. Format:
{
  "tool": "<tool_name>",
  "parameters": {<tool_parameters>},
  "reasoning": "<brief clinical reasoning>"
}
"""

# ---------------------------------------------------------------------------
# Tool schema builder (for LLM function calling)
# ---------------------------------------------------------------------------

def build_tool_schemas() -> list:
    """Build OpenAI-compatible tool schemas from TOOL_DEFINITIONS."""
    schemas = []
    for name, defn in TOOL_DEFINITIONS.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": defn["description"],
                "parameters": {
                    "type": "object",
                    "properties": defn.get("parameters", {}),
                    "required": list(defn.get("parameters", {}).keys()),
                },
            }
        })
    return schemas


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent_episode(
    env: VetTriageEnv,
    client,
    backend: str,
    task_id: str,
    model: str,
    seed: Optional[int],
    verbose: bool = True,
) -> dict:
    """Run one episode and return results."""

    obs = env.reset(task_id=task_id, seed=seed)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"New patient case:\n"
                f"Species: {obs.species} | Breed: {obs.breed} | "
                f"Age: {obs.age_years}yr | Sex: {obs.sex} | Weight: {obs.weight_kg}kg\n\n"
                f"Presenting complaint: {obs.presenting_complaint}\n\n"
                f"You are in phase: {obs.phase}\n"
                f"Available tools: {', '.join(obs.available_tools)}\n\n"
                f"Begin your assessment."
            ),
        },
    ]

    total_reward = 0.0
    done = False
    steps = 0
    info = {}

    while not done and steps < 80:
        # Build current state message
        state_summary = _format_observation(obs)

        if steps > 0:
            messages.append({
                "role": "user",
                "content": f"Step {obs.step} — Current state:\n{state_summary}"
            })

        # Get agent response
        try:
            raw = _call_llm(client, backend, model, messages)
            action_dict = json.loads(raw)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"  [Step {steps}] JSON parse error: {e}")
            action_dict = {
                "tool": "check_vitals",
                "parameters": {"systems": ["all"]},
                "reasoning": "fallback"
            }
        except Exception as e:
            if verbose:
                print(f"  [Step {steps}] API error: {e}")
            break

        action = Action(
            tool=action_dict.get("tool", "check_vitals"),
            parameters=action_dict.get("parameters", {}),
            reasoning=action_dict.get("reasoning"),
        )

        if verbose:
            print(f"  [Step {steps}] {action.tool}({json.dumps(action.parameters, separators=(',', ':'))[:60]})"
                  f" — {(action.reasoning or '')[:50]}")

        # Add assistant response to history
        messages.append({"role": "assistant", "content": raw})

        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

        if verbose and reward.message:
            print(f"           → {reward.message[:80]} | r={reward.value:+.3f}")

        if obs.events:
            for evt in obs.events:
                if verbose:
                    print(f"  [EVENT] {evt}")

    return {
        "task_id": task_id,
        "grade": info.get("grade", 0.0),
        "passed": info.get("passed", False),
        "total_reward": round(total_reward, 3),
        "steps": steps,
        "terminal_reason": info.get("terminal_reason", "unknown"),
        "breakdown": info.get("grade_breakdown", {}),
        "feedback": info.get("grade_feedback", []),
    }


def _format_observation(obs) -> str:
    """Format observation as compact text for LLM context."""
    lines = [
        f"Phase: {obs.phase} (step {obs.phase_step}/{obs.phase_step_limit})",
    ]
    if obs.vitals:
        v = obs.vitals
        lines.append(
            f"Vitals: HR={v.get('heart_rate','?')} | RR={v.get('respiratory_rate','?')} | "
            f"SpO2={v.get('spo2','?')}% | BP={v.get('systolic_bp','?')} | "
            f"MM={v.get('mucous_membrane_color','?')} | Mentation={v.get('mentation','?')}"
        )
    if obs.physical_exam_findings:
        for region, finding in obs.physical_exam_findings.items():
            lines.append(f"Exam [{region}]: {finding}")
    if obs.lab_results:
        for panel, result in obs.lab_results.items():
            if isinstance(result, dict):
                interp = result.get("interpretation", "")
                lines.append(f"Lab [{panel}]: {interp}")
    if obs.imaging_results:
        for key, result in obs.imaging_results.items():
            if isinstance(result, dict):
                interp = result.get("interpretation", "")
                lines.append(f"Imaging [{key}]: {interp}")
    if obs.pending_results:
        for p in obs.pending_results:
            lines.append(f"Pending: {p.panel_or_modality} (ETA {p.eta_steps} steps, job={p.job_id})")
    if obs.owner_contact_established and obs.budget_limit:
        lines.append(f"Budget: £{obs.budget_spent:.0f} spent / £{obs.budget_limit:.0f} limit")
    if obs.specialist_opinion:
        lines.append(f"Specialist: {obs.specialist_opinion[:100]}")
    if obs.events:
        for evt in obs.events:
            lines.append(f"! EVENT: {evt}")
    lines.append(f"Available tools: {', '.join(obs.available_tools)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VetTriageEnv baseline inference")
    parser.add_argument("--backend", default="openai", choices=["openai", "anthropic"],
                        help="LLM backend to use")
    parser.add_argument("--model", default=None,
                        help="Model name (default: gpt-4o-mini for openai, claude-haiku-4-5-20251001 for anthropic)")
    parser.add_argument("--task", default=None, help="Specific task ID (or all if omitted)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--runs", type=int, default=1, help="Runs per task")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-level output")
    args = parser.parse_args()

    # Default model per backend
    if args.model is None:
        args.model = "claude-sonnet-4-6" if args.backend == "anthropic" else "gpt-4o-mini"

    client = _make_client(args.backend)
    env = VetTriageEnv(max_total_steps=80)

    tasks_to_run = [args.task] if args.task else list_tasks()
    all_results = []

    print(f"\nVetTriageEnv Baseline — backend: {args.backend} | model: {args.model}")
    print("=" * 60)

    for task_id in tasks_to_run:
        task_results = []
        print(f"\nTask: {task_id}")
        print("-" * 40)

        for run in range(args.runs):
            print(f"  Run {run + 1}/{args.runs}:")
            result = run_agent_episode(
                env, client, args.backend, task_id,
                model=args.model,
                seed=args.seed + run,
                verbose=not args.quiet,
            )
            task_results.append(result)
            print(f"  → Grade: {result['grade']:.3f} | Passed: {result['passed']} | "
                  f"Steps: {result['steps']} | Reward: {result['total_reward']:+.3f}")
            if result["feedback"]:
                print(f"  Feedback: {'; '.join(result['feedback'][:3])}")

        avg_grade = sum(r["grade"] for r in task_results) / len(task_results)
        avg_reward = sum(r["total_reward"] for r in task_results) / len(task_results)
        pass_rate = sum(1 for r in task_results if r["passed"]) / len(task_results)

        print(f"\n  Summary [{task_id}]: avg_grade={avg_grade:.3f} | "
              f"pass_rate={pass_rate:.0%} | avg_reward={avg_reward:+.3f}")
        all_results.extend(task_results)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("-" * 40)
    for task_id in tasks_to_run:
        task_r = [r for r in all_results if r["task_id"] == task_id]
        avg = sum(r["grade"] for r in task_r) / len(task_r)
        print(f"  {task_id:30s}: {avg:.3f}")
    overall = sum(r["grade"] for r in all_results) / len(all_results)
    print(f"  {'OVERALL':30s}: {overall:.3f}")

    # Save results
    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "seed": args.seed,
            "results": all_results,
            "summary": {
                task_id: {
                    "avg_grade": sum(r["grade"] for r in all_results if r["task_id"] == task_id) /
                                 max(1, sum(1 for r in all_results if r["task_id"] == task_id))
                }
                for task_id in tasks_to_run
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

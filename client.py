"""
VetTriageEnv Client — OpenEnv-compliant client for the veterinary triage environment.

Provides a Python client interface for interacting with the VetTriageEnv.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from vettriagevenv.env import VetTriageEnv
from vettriagevenv.models import Action, Observation


class VetTriageClient:
    """
    Client for the VetTriageEnv.

    Provides methods to interact with the veterinary triage environment.
    """

    def __init__(self, task: str = "easy_gdv"):
        """
        Initialize the client.

        Args:
            task: The task to run. One of: easy_gdv, medium_hcm_cat, hard_polytrauma
        """
        self.env = VetTriageEnv()
        self.task = task

    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset the environment and return the initial observation.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            The initial observation
        """
        return self.env.reset(task_id=self.task, seed=seed)

    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: The action to take (Action with tool, parameters, reasoning)

        Returns:
            observation, reward_value, done, info
        """
        obs, reward, done, info = self.env.step(action)
        return obs, reward.value, done, info

    def state(self) -> dict:
        """
        Return the full internal state (ground truth, for logging/debugging).

        Returns:
            Dict with internal_state, meta, action_log, cumulative_reward
        """
        return self.env.state()


# For OpenEnv compatibility
def create_client(**kwargs) -> VetTriageClient:
    """
    Create a VetTriageClient instance.

    Args:
        **kwargs: Arguments to pass to VetTriageClient (e.g. task="medium_hcm_cat")

    Returns:
        VetTriageClient instance
    """
    return VetTriageClient(**kwargs)

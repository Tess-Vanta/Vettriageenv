"""
VetTriageEnv Client — OpenEnv-compliant client for the veterinary triage environment.

Provides a Python client interface for interacting with the VetTriageEnv.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Import the environment
from vettriagevenv.env import VetTriageEnv
from vettriagevenv.models import Action, Observation, State


class VetTriageClient:
    """
    Client for the VetTriageEnv.

    Provides methods to interact with the veterinary triage environment.
    """

    def __init__(self, task: str = "easy_gdv"):
        """
        Initialize the client.

        Args:
            task: The task to run (default: easy_gdv)
        """
        self.env = VetTriageEnv(task=task)
        self.state = None

    def reset(self) -> State:
        """
        Reset the environment and return the initial state.

        Returns:
            The initial state
        """
        self.state = self.env.reset()
        return self.state

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: The action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        return self.env.step(action)

    def render(self) -> str:
        """
        Render the current state.

        Returns:
            String representation of the current state
        """
        return self.env.render()

    def close(self):
        """
        Close the environment.
        """
        self.env.close()


# For OpenEnv compatibility
def create_client(**kwargs) -> VetTriageClient:
    """
    Create a VetTriageClient instance.

    Args:
        **kwargs: Arguments to pass to VetTriageClient

    Returns:
        VetTriageClient instance
    """
    return VetTriageClient(**kwargs)
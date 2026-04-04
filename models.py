"""Data models for the OpenClean environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------- ACTION ----------------
class OpenCleanAction(Action):
    """Action for the OpenClean environment."""

    action: Literal["REMOVE_DUPLICATES", "FILL_MISSING", "FIX_FORMAT"] = Field(
        ...,
        description="Data-cleaning operation to apply to the dataset.",
    )


# ---------------- OBSERVATION ----------------
class OpenCleanObservation(Observation):
    """Observation returned after each step."""

    # ---------- Core ----------
    message: str = Field(default="", description="Environment status message.")

    # ---------- Dataset State ----------
    rows: int = Field(default=0, description="Number of rows in dataset.")
    columns: List[str] = Field(default_factory=list, description="Column names.")

    missing_values: int = Field(default=0, description="Count of missing values.")

    duplicate_rows: int = Field(default=0, description="Number of duplicate rows.")

    invalid_emails: int = Field(
        default=0,
        description="Number of invalid email entries.",
    )

    sample: Dict[str, Dict[int, Any]] = Field(
        default_factory=dict,
        description="Preview of dataset.",
    )

    # ---------- RL Signals ----------
    reward: Optional[float] = Field(
        default=None,
        description="Reward received after action.",
    )

    done: bool = Field(default=False, description="Whether episode is finished.")

    # ---------- Task Info ----------
    goal: str = Field(default="", description="Task goal for the agent.")

    # ---------- Evaluation ----------
    final_score: Optional[float] = Field(
        default=None,
        description="Final evaluation score (0–1).",
    )

    # ---------- Metadata ----------
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional info like step number, task type, etc.",
    )


# ---------------- ALIASES ----------------
KernelAction = OpenCleanAction
KernelObservation = OpenCleanObservation
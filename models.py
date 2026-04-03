from pydantic import BaseModel
from typing import Dict, Any

class Action(BaseModel):
    action_type: str


class OpenCleanAction(Action):
    """Compatibility action model exported by kernel_env package."""

    pass

class Observation(BaseModel):
    data_preview: Dict[str, Any]


class OpenCleanObservation(Observation):
    """Compatibility observation model exported by kernel_env package."""

    pass

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
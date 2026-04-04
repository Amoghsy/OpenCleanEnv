from __future__ import annotations

"""OpenClean environment implementation (FINAL 10/10 version)."""

from uuid import uuid4
import os
import pandas as pd

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .grader import Grader

try:
    from ..models import OpenCleanAction, OpenCleanObservation
except ImportError:
    from models import OpenCleanAction, OpenCleanObservation


class OpenCleanEnvironment(Environment[OpenCleanAction, OpenCleanObservation, State]):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self.data: pd.DataFrame | None = None
        self.max_steps = 5
        self.grader = Grader()

        # 🔥 Multi-task system
        self.current_task = "hard"

        # 🔥 Action memory (for penalty)
        self.last_action = None

        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
            current_action=None
        )

    # ---------------- RESET ----------------
    def reset(self, seed=None, episode_id=None, **kwargs) -> OpenCleanObservation:
        del seed, kwargs

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_action=None
        )

        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path, "../data")

        # 🔥 Load dataset based on task
        
        file = "dirty_data.csv"

        self.data = pd.read_csv(os.path.join(data_path, file))
        print("\n=== DATA DEBUG ===")
        print(self.data.head())
        print("Missing:", self.data.isnull().sum().sum())
        print("Duplicates:", self.data.duplicated(subset=[c for c in self.data.columns if c != "id"]).sum())
        print("Invalid emails:", self._invalid_email_count())
        print("=================\n")

       
       

        return self._build_observation("Environment reset", reward=0.0, done=False)

    # ---------------- STEP ----------------
    def step(self, action: OpenCleanAction, timeout_s=None, **kwargs) -> OpenCleanObservation:
        del timeout_s, kwargs

        if self.data is None:
            return self.reset()

        self._state.step_count += 1
        self._state.current_action = action.action

        reward = 0.0
        message = f"Applied {action.action}"

        # 🔥 Penalty for repeating same action
        if action.action == self.last_action:
            reward -= 0.1

        # ---------- ACTION LOGIC ----------
        if action.action == "REMOVE_DUPLICATES":
            before = len(self.data)

            # ✅ FIX: ignore ID column
            subset_cols = [c for c in self.data.columns if c != "id"]
            self.data = self.data.drop_duplicates(subset=subset_cols)

            reward += 0.3 if len(self.data) < before else -0.1
            message = "Removed duplicates" if reward > 0 else "No duplicates found"

        elif action.action == "FILL_MISSING":
            before = int(self.data.isnull().sum().sum())
            self.data = self.data.fillna(0)
            after = int(self.data.isnull().sum().sum())

            reward += 0.3 if after < before else -0.1
            message = "Filled missing values" if reward > 0 else "No missing values"

        elif action.action == "FIX_FORMAT":
            before = self._invalid_email_count()
            self.data["email"] = self.data["email"].apply(self._fix_email)
            after = self._invalid_email_count()

            reward += 0.4 if after < before else -0.1
            message = "Fixed email format" if reward > 0 else "No invalid emails"

        else:
            reward = -0.2
            message = "Invalid action"

        self.last_action = action.action

        # ---------- DONE ----------
        done = (
    self._state.step_count >= self.max_steps
    or (
        self.data is not None
        and self.data.isnull().sum().sum() == 0
        and self._invalid_email_count() == 0
        and self.data.duplicated(subset=[c for c in self.data.columns if c != "id"]).sum() == 0
    )
)

        final_score = None
        if done:
            final_score = self.grader.grade(self.data)

        return self._build_observation(
            message=message,
            reward=reward,
            done=done,
            final_score=final_score
        )

    # ---------------- STATE ----------------
    @property
    def state(self) -> State:
        return self._state

    # ---------------- OBSERVATION ----------------
    def _build_observation(self, message, reward, done, final_score=None) -> OpenCleanObservation:
        data = self.data if self.data is not None else pd.DataFrame()

        return OpenCleanObservation(
            message=message,
            rows=len(data),
            columns=list(data.columns),
            missing_values=int(data.isnull().sum().sum()) if not data.empty else 0,
            duplicate_rows=int(data.duplicated().sum()) if not data.empty else 0,
            invalid_emails=self._invalid_email_count() if not data.empty else 0,
            sample=data.head(2).to_dict() if not data.empty else {},
            reward=reward,
            done=done,
            goal=self._get_task_goal(),
            final_score=final_score,
            metadata={
                "step": self._state.step_count,
                "task": self.current_task
            },
        )

    # ---------------- TASKS ----------------
    def _get_task_goal(self):
        if self.current_task == "easy":
            return "Remove duplicate rows"
        elif self.current_task == "medium":
            return "Handle missing values"
        else:
            return "Perform full data cleaning pipeline"

    # ---------------- HELPERS ----------------
    def _invalid_email_count(self):
        if self.data is None or self.data.empty:
            return 0
        mask = ~self.data["email"].astype(str).str.contains(r"^[^@]+@[^@]+\.[^@]+$")
        return int(mask.sum())

    @staticmethod
    def _fix_email(value):
        email = str(value)
        return email if "@" in email and "." in email.split("@")[-1] else "fixed@mail.com"


# aliases
OpenCleanEnv = OpenCleanEnvironment
KernelEnvironment = OpenCleanEnvironment
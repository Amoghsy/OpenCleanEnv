from __future__ import annotations

"""OpenClean environment implementation — FINAL FIXED VERSION."""

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

        self._data_path: str | None = None
        self.data: pd.DataFrame | None = None

        # Track which actions have been successfully applied this episode
        self._actions_applied: set[str] = set()

        self.max_steps = 5
        self.grader = Grader()

        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
            current_action=None
        )

    # ─────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────
    def reset(self, seed=None, episode_id=None, **kwargs) -> OpenCleanObservation:
        del seed, kwargs

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_action=None
        )

        # Clear applied-actions tracker for new episode
        self._actions_applied = set()

        # Load fresh dirty data
        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path, "../data")
        csv_path = os.path.join(data_path, "dirty_data.csv")

        self._data_path = csv_path
        self.data = pd.read_csv(csv_path)

        return self._build_observation(
            "Environment reset",
            reward=0.0,
            done=False,
            final_score=self.grader.grade(self.data),
        )

    # ─────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────
    def step(self, action: OpenCleanAction, timeout_s=None, **kwargs) -> OpenCleanObservation:
        del timeout_s, kwargs

        # Safety: restore data if None (e.g. server restart mid-session)
        if self.data is None:
            if self._data_path and os.path.exists(self._data_path):
                self.data = pd.read_csv(self._data_path)
            else:
                return self.reset()

        self._state.step_count += 1
        self._state.current_action = action.action

        reward = 0.0
        message = f"Applied {action.action}"

        # ──────────────────────────────────────
        # REMOVE DUPLICATES
        # ──────────────────────────────────────
        if action.action == "REMOVE_DUPLICATES":
            subset_cols = [c for c in self.data.columns if c != "id"]
            before = len(self.data)

            # Normalize object columns before comparing
            for col in subset_cols:
                if self.data[col].dtype == object:
                    self.data[col] = self.data[col].astype(str).str.strip()

            self.data = self.data.drop_duplicates(subset=subset_cols).reset_index(drop=True)
            after = len(self.data)

            if after < before:
                reward = 0.3
                message = f"Removed {before - after} duplicate rows"
                self._actions_applied.add("REMOVE_DUPLICATES")
            else:
                reward = -0.1
                message = "No duplicates found"

        # ──────────────────────────────────────
        # FILL MISSING
        # ──────────────────────────────────────
        elif action.action == "FILL_MISSING":
            before = int(self.data.isnull().sum().sum())

            for col in self.data.columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    self.data[col] = self.data[col].fillna(0)
                else:
                    self.data[col] = self.data[col].fillna("")

            after = int(self.data.isnull().sum().sum())

            if after < before:
                reward = 0.3
                message = f"Filled {before - after} missing values"
                self._actions_applied.add("FILL_MISSING")
            else:
                reward = -0.1
                message = "No missing values found"

        # ──────────────────────────────────────
        # FIX FORMAT (emails)
        # ──────────────────────────────────────
        elif action.action == "FIX_FORMAT":
            before = self._invalid_email_count()
            self.data["email"] = self.data["email"].apply(self._fix_email)
            after = self._invalid_email_count()

            if after < before:
                reward = 0.4
                message = f"Fixed {before - after} invalid email(s)"
                self._actions_applied.add("FIX_FORMAT")
            else:
                reward = -0.1
                message = "No invalid emails found"

        else:
            reward = -0.2
            message = "Invalid action"

        # ──────────────────────────────────────
        # DONE condition
        # done = True when:
        #   1. All 3 actions successfully applied, OR
        #   2. Dataset is genuinely clean, OR
        #   3. max_steps reached
        # ──────────────────────────────────────
        all_three_done = self._actions_applied >= {
            "REMOVE_DUPLICATES", "FILL_MISSING", "FIX_FORMAT"
        }

        missing_count = int(self.data.isnull().sum().sum())
        email_count   = self._invalid_email_count()
        dup_count     = int(
            self.data.duplicated(
                subset=[c for c in self.data.columns if c != "id"]
            ).sum()
        )
        dataset_clean = (missing_count == 0 and email_count == 0 and dup_count == 0)

        done = (
            self._state.step_count >= self.max_steps
            or all_three_done
            or dataset_clean
        )

        # Always provide a score snapshot so final_score is never null.
        final_score = self.grader.grade(self.data)

        return self._build_observation(
            message=message,
            reward=reward,
            done=done,
            final_score=final_score,
        )

    # ─────────────────────────────────────────
    # STATE
    # ─────────────────────────────────────────
    @property
    def state(self) -> State:
        return self._state

    # ─────────────────────────────────────────
    # OBSERVATION BUILDER
    # ─────────────────────────────────────────
    def _build_observation(self, message, reward, done, final_score=None) -> OpenCleanObservation:
        data = self.data if self.data is not None else pd.DataFrame()

        return OpenCleanObservation(
            message=message,
            rows=len(data),
            columns=list(data.columns),
            missing_values=int(data.isnull().sum().sum()) if not data.empty else 0,
            duplicate_rows=int(
                data.duplicated(
                    subset=[c for c in data.columns if c != "id"]
                ).sum()
            ) if not data.empty else 0,
            invalid_emails=self._invalid_email_count() if not data.empty else 0,
            sample=data.head(2).to_dict() if not data.empty else {},
            reward=reward,
            done=done,
            goal="Perform full data cleaning pipeline: remove duplicates, fill missing values, fix email formats",
            final_score=final_score,
            metadata={
                "step": self._state.step_count,
                "actions_applied": list(self._actions_applied),
            },
        )

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────
    def _invalid_email_count(self) -> int:
        if self.data is None or self.data.empty:
            return 0
        mask = ~self.data["email"].astype(str).str.contains(
            r"^[^@]+@[^@]+\.[^@]+$", na=False, regex=True
        )
        return int(mask.sum())

    @staticmethod
    def _fix_email(value: str) -> str:
        email = str(value).strip()
        if "@" in email and "." in email.split("@")[-1]:
            return email
        return "fixed@mail.com"


# Aliases expected by app.py
OpenCleanEnv      = OpenCleanEnvironment
KernelEnvironment = OpenCleanEnvironment
import pandas as pd
import numpy as np


class OpenCleanEnv:
    def __init__(self):
        self.data = None
        self.steps = 0
        self.max_steps = 5

    # ------------------------
    # RESET
    # ------------------------
    def reset(self):
        self.steps = 0

        # Create realistic messy dataset
        self.data = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Charlie"],
            "age": [25, np.nan, 30, 30],
            "salary": [50000, None, 60000, 60000],
            "email": ["alice@mail.com", "bob@mail", "charlie@mail.com", "charlie@mail.com"]
        })

        return {
            "observation": {
                "message": "Environment reset",
                "rows": len(self.data),
                "missing_values": int(self.data.isnull().sum().sum())
            }
        }

    # ------------------------
    # STEP
    # ------------------------
    def step(self, action):
        self.steps += 1
        reward = 0.0

        # ---------- ACTIONS ----------
        if action == "REMOVE_DUPLICATES":
            before = len(self.data)
            self.data = self.data.drop_duplicates()
            after = len(self.data)

            if after < before:
                reward = 0.3
            else:
                reward = -0.1

        elif action == "FILL_MISSING":
            missing_before = self.data.isnull().sum().sum()
            self.data = self.data.fillna(0)
            missing_after = self.data.isnull().sum().sum()

            if missing_after < missing_before:
                reward = 0.3
            else:
                reward = -0.1

        elif action == "FIX_FORMAT":
            # Fix invalid emails
            def fix_email(e):
                return e if "@" in str(e) and "." in str(e) else "fixed@mail.com"

            self.data["email"] = self.data["email"].apply(fix_email)
            reward = 0.4

        else:
            reward = -0.2  # invalid action penalty

        # ---------- DONE CONDITION ----------
        done = self.steps >= self.max_steps

        # ---------- OBSERVATION ----------
        observation = {
            "rows": len(self.data),
            "missing_values": int(self.data.isnull().sum().sum()),
            "sample": self.data.head(2).to_dict()
        }

        return {
            "observation": observation,
            "reward": float(round(reward, 2)),
            "done": bool(done),
            "info": {
                "step": self.steps
            }
        }

    # ------------------------
    # STATE
    # ------------------------
    def state(self):
        return {
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "missing_values": int(self.data.isnull().sum().sum())
        }
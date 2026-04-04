import os

import pandas as pd


class Grader:
    """Simple score function comparing current data against clean reference data."""

    def __init__(self):
        # Load ground truth from CSV
        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path, "../data")

        self.clean_data = pd.read_csv(os.path.join(data_path, "clean_data.csv"))

    def grade(self, df: pd.DataFrame) -> float:
        """Return a bounded score in [0, 1] based on cleaning quality."""
        score = 0.0

        # ---------- SAFETY ----------
        if df is None or df.empty:
            return 0.0

        # Ensure same column order
        try:
            df = df[self.clean_data.columns]
        except Exception:
            return 0.0

        # ---------- 1. Row count ----------
        if len(df) == len(self.clean_data):
            score += 0.25
        else:
            score += 0.25 * (
                min(len(df), len(self.clean_data)) / max(len(df), len(self.clean_data))
            )

        # ---------- 2. Missing values ----------
        missing = df.isnull().sum().sum()
        total_cells = df.size

        if total_cells > 0:
            score += 0.25 * (1 - (missing / total_cells))

        # ---------- 3. Email validity ----------
        if "email" in df.columns:
            valid_mask = df["email"].astype(str).str.contains(r"^[^@]+@[^@]+\.[^@]+$")
            score += 0.2 * valid_mask.mean()

        # ---------- 4. Data similarity ----------
        try:
            df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)
            clean_sorted = self.clean_data.sort_values(
                by=list(self.clean_data.columns)
            ).reset_index(drop=True)

            matches = (df_sorted == clean_sorted).sum().sum()
            total = df_sorted.size

            if total > 0:
                score += 0.3 * (matches / total)
        except Exception:
            pass

        # ---------- CLAMP ----------
        score = max(0.0, min(1.0, score))

        return round(score, 3)
import pandas as pd

class Grader:
    def __init__(self):
        # Ground truth
        self.clean_data = pd.DataFrame({
            "name": ["A", "B", "C"],
            "age": [25, 0, 30],
            "salary": [50000, 60000, 60000]
        })

    def grade(self, agent_data):
        df = pd.DataFrame(agent_data)

        score = 0.0

        # check duplicates removed
        if len(df) == len(self.clean_data):
            score += 0.3

        # check missing handled
        if df.isnull().sum().sum() == 0:
            score += 0.3

        # check approximate correctness
        if df.equals(self.clean_data):
            score += 0.4

        return round(score, 2)
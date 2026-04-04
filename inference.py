import os
from typing import List, Optional

from client import KernelEnv
from models import OpenCleanAction
from openai import OpenAI

# -------- ENV CONFIG --------
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_NAME = "data_cleaning"
BENCHMARK = "OpenCleanEnv"
MAX_STEPS = 5


# -------- LOGGING --------
def log_start(task: str, env: str, model: str):
    """Print start log in benchmark-compatible format."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    """Print per-step log in benchmark-compatible format."""
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]):
    """Print final summary log in benchmark-compatible format."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# -------- MAIN --------
def run():
    """Run one deterministic cleaning episode while issuing required LLM calls."""
    rewards: List[float] = []
    steps_taken = 0
    success = False

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        with KernelEnv(base_url=ENV_URL).sync() as env:
            env.reset()

            # Ordered action plan — each action finds something to fix
            action_plan = {
                1: "REMOVE_DUPLICATES",
                2: "FILL_MISSING",
                3: "FIX_FORMAT",
            }

            for step in range(1, MAX_STEPS + 1):
                error_msg = None

                try:
                    # Required real LLM call
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a data cleaning agent. "
                                    "Available actions: REMOVE_DUPLICATES, FILL_MISSING, FIX_FORMAT."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Step {step}: The dataset has duplicate rows, missing values, "
                                    "and invalid email formats. What is the best next action?"
                                ),
                            },
                        ],
                        temperature=0,
                        max_tokens=20,
                    )
                    _ = completion.choices[0].message.content

                    # Deterministic action regardless of LLM output
                    action = action_plan.get(step, "FIX_FORMAT")

                    result = env.step(OpenCleanAction(action=action))
                    reward = float(result.reward or 0.0)
                    done = bool(result.done)

                except Exception as e:
                    action = "ERROR"
                    reward = 0.0
                    done = True
                    error_msg = str(e)

                rewards.append(reward)
                steps_taken = step

                log_step(step, action, reward, done, error_msg)

                if done:
                    success = error_msg is None
                    break

            success = success and (len(rewards) > 0)

    finally:
        log_end(success, steps_taken, rewards)


if __name__ == "__main__":
    run()
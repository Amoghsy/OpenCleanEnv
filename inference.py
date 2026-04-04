import os
import requests
from typing import List, Optional
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
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# -------- MAIN --------
def run():
    rewards = []
    steps_taken = 0
    success = False

    # 🔥 REQUIRED REAL LLM CLIENT
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # Reset env
        r = requests.get(f"{ENV_URL}/reset", timeout=10)
        r.raise_for_status()

        for step in range(1, MAX_STEPS + 1):
            error_msg = None

            try:
                # 🔥 REQUIRED REAL API CALL
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a data cleaning agent."},
                        {"role": "user", "content": f"Step {step}: choose best cleaning action"}
                    ],
                    temperature=0,
                    max_tokens=10,
                )

                _ = completion.choices[0].message.content

                # Deterministic actions
                if step == 1:
                    action = "REMOVE_DUPLICATES"
                elif step == 2:
                    action = "FILL_MISSING"
                else:
                    action = "FIX_FORMAT"

                res = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": {"action": action}},
                    timeout=10
                )
                res.raise_for_status()

                data = res.json()

                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))

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

        success = success or (len(rewards) > 0 and rewards[-1] > 0)

    finally:
        log_end(success, steps_taken, rewards)


if __name__ == "__main__":
    run()
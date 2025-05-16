import random
from tqdm import tqdm
import pandas as pd
import gymnasium as gym

from formatters import get_formatter
from processors import get_processor

from gymnasium.envs.registration import register
from custom.custom_minigrid_envs.goto import CustomGoToRedBallEnv

# ──────────────── CONFIG ────────────────

ALL_ENVS = [
    "CustomBabyAI-GoToRedBall-Small-4Dists-v0", "CustomBabyAI-GoToRedBall-Small-5Dists-v0",
    "CustomBabyAI-GoToRedBall-Small-6Dists-v0", "CustomBabyAI-GoToRedBall-Small-7Dists-v0",
    "CustomBabyAI-GoToRedBall-Medium-20Dists-v0", "CustomBabyAI-GoToRedBall-Medium-40Dists-v0",
    "CustomBabyAI-GoToRedBall-Medium-50Dists-v0", "CustomBabyAI-GoToRedBall-Medium-60Dists-v0",
    "CustomBabyAI-GoToRedBall-Large-60Dists-v0", "CustomBabyAI-GoToRedBall-Large-80Dists-v0",
    "CustomBabyAI-GoToRedBall-Large-100Dists-v0", "CustomBabyAI-GoToRedBall-Large-120Dists-v0",
    "CustomBabyAI-GoToRedBall-Ultra-120Dists-v0", "CustomBabyAI-GoToRedBall-Ultra-140Dists-v0",
    "CustomBabyAI-GoToRedBall-Ultra-160Dists-v0", "CustomBabyAI-GoToRedBall-Ultra-180Dists-v0"
]

ALL_SEEDS = range(1, 501)

formatter = get_formatter("structured")
processor = get_processor("omniscient_babyai_bot")

# ──────────────── UTILITIES ────────────────

def make_env(env_name, seed):
    env = gym.make(env_name, tile_size=32, render_mode='rgb_array')
    env.reset(seed=seed)
    return env

def get_red_ball_pos(env):
    for i in range(env.unwrapped.grid.width):
        for j in range(env.unwrapped.grid.height):
            cell = env.unwrapped.grid.get(i, j)
            if cell is not None and cell.type == 'ball' and cell.color == 'red':
                return (i, j)
    raise ValueError("Red ball not found (unexpected condition).")

def generate_example(env_name, seed):
    env = make_env(env_name, seed)
    
    description = formatter.format(env)
    agent_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
    agent_dir = int(env.unwrapped.agent_dir)
    initial_state = (agent_pos, agent_dir)

    red_ball_pos = get_red_ball_pos(env)
    target_subgoal = red_ball_pos

    action_sequence = processor.process(env_name, seed, task="plan")

    return {
        "env_name": env_name,
        "seed": seed,
        "env_description": description,
        "initial_state": str(initial_state),
        "target_subgoal": str(target_subgoal),
        "expert_action_sequence": action_sequence
    }

# ──────────────── MAIN GENERATION ────────────────

records = []
for env_name in tqdm(ALL_ENVS, desc="Generating PLAN examples"):
    for seed in ALL_SEEDS:
        try:
            rec = generate_example(env_name, seed)
            records.append(rec)
        except Exception as e:
            print(f"Failed on {env_name}, seed {seed}: {e}")

if not records:
    raise RuntimeError("No records generated! Check logic.")

# ──────────────── SAVE CSV ────────────────

df = pd.DataFrame.from_records(records)
output_path = "LLM_BABYBENCH_Plan.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} examples to {output_path}")


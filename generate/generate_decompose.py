import pandas as pd
from tqdm import tqdm
import gymnasium as gym

from formatters import get_formatter
from custom.custom_babyai_bots.omniscient_fixed_plan_baby_ai_bot import BabyAIBot

# ──────────────── CONFIG ────────────────

ALL_ENVS = [
    "BabyAI-GoToObj-v0", "BabyAI-GoToRedBallGrey-v0", "BabyAI-GoToRedBall-v0", 
    "BabyAI-GoToLocal-v0", "BabyAI-PutNextLocal-v0", "BabyAI-PickupLoc-v0", 
    "BabyAI-GoToObjMaze-v0", "BabyAI-GoTo-v0", "BabyAI-Pickup-v0", 
    "BabyAI-UnblockPickup-v0", "BabyAI-Open-v0", "BabyAI-Synth-v0", 
    "BabyAI-SynthLoc-v0", "BabyAI-GoToSeq-v0", "BabyAI-SynthSeq-v0", "BabyAI-BossLevel-v0"
]

ALL_SEEDS = range(1, 501)

formatter = get_formatter("structured")

# ──────────────── UTILITIES ────────────────

def make_env(env_name, seed):
    env = gym.make(env_name, tile_size=32, render_mode='rgb_array')
    env.reset(seed=seed)
    return env

def generate_example(env_name, seed):
    env = make_env(env_name, seed)

    description = formatter.format(env)

    agent_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
    agent_dir = int(env.unwrapped.agent_dir)
    initial_state = str((agent_pos, agent_dir))

    mission = str(env.unwrapped.mission)

    # Use fixed plan bot to solve and retrieve help count
    bot = BabyAIBot(env, [])
    bot.provide_initial_subgoals()
    for i in range(1000):
        action = bot.replan()
        _, _, done, truncated, _ = env.step(action)
        if done:
            break
        bot.stack = bot.stack
    help_count = bot.return_help_count()


    return {
        "env_name": env_name,
        "seed": seed,
        "env_description": description,
        "initial_state": initial_state,
        "mission": mission,
        "help_count": help_count,
    }

# ──────────────── MAIN GENERATION ────────────────

records = []
for env_name in tqdm(ALL_ENVS, desc="Generating Decompose examples"):
    for seed in ALL_SEEDS:
        try:
            rec = generate_example(env_name, seed)
            records.append(rec)
        except Exception as e:
            print(f"Failed on {env_name}, seed {seed}: {e}")

if not records:
    raise RuntimeError("No records generated! Check your generation functions.")

# ──────────────── SAVE ────────────────

df = pd.DataFrame.from_records(records)
output_path = "LLM_BABYBENCH_Decompose.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} examples to {output_path}")


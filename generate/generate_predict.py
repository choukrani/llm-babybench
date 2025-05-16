import random
from tqdm import tqdm
import pandas as pd
import gymnasium as gym

from formatters import get_formatter
from processors import get_processor
from evaluators.utils import str_action_seq_to_int

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
processor = get_processor("omniscient_babyai_bot")

# ──────────────── UTILITIES ────────────────

def make_env(env_name, seed):
    """Create and reset a BabyAI-like environment."""
    env = gym.make(env_name, tile_size=32, render_mode='rgb_array')
    env.reset(seed=seed)
    return env

def clone_and_execute(env, action_sequence):
    """
    Deep-copy the env, convert the comma-separated action_sequence to ints,
    step through them, and return the final (pos_x, pos_y), dir tuple.
    """
    from copy import deepcopy
    cloned_env = deepcopy(env)
    int_action_seq = str_action_seq_to_int(action_sequence)
    for action in int_action_seq:
        _, _, terminated, truncated, _ = cloned_env.step(action)
        if terminated or truncated:
            break
    final_pos = tuple(int(x) for x in cloned_env.unwrapped.agent_pos)
    final_dir = int(cloned_env.unwrapped.agent_dir)
    return (final_pos, final_dir)

def generate_example(env_name, seed):
    """
    Generate a single example:
      - env_description via formatter
      - initial_state as ((x,y), dir)
      - action_sequence (comma-separated str)
      - target_state as ((x,y), dir)
    """
    env = make_env(env_name, seed)
    description = formatter.format(env)

    init_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
    init_dir = int(env.unwrapped.agent_dir)
    initial_state = (init_pos, init_dir)

    action_sequence = processor.process(env_name, seed, task="predict")
    target_state = clone_and_execute(env, action_sequence)

    return {
        "env_name":        env_name,
        "seed":            seed,
        "env_description": description,
        # stringify the tuple structures
        "initial_state":   str(initial_state),
        "action_sequence": action_sequence,
        "target_state":    str(target_state),
    }

# ──────────────── MAIN GENERATION ────────────────

records = []
for env_name in tqdm(ALL_ENVS, desc="Generating examples"):
    for seed in ALL_SEEDS:
        try:
            rec = generate_example(env_name, seed)
            records.append(rec)
        except Exception as e:
            print(f"Failed on {env_name}, seed {seed}: {e}")

if not records:
    raise RuntimeError("No records generated! Check your generation functions.")

# ──────────────── SAVE TO CSV ────────────────

df = pd.DataFrame.from_records(records)
output_path = "LLM_BABYBENCH_Predict.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} examples to {output_path}")


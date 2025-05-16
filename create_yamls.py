import re
import os
import yaml
from itertools import product

# Create output directory
output_dir = "configs"
os.makedirs(output_dir, exist_ok=True)

# Define the values for each parameter
llms = [
    {"name": "openai", "model": "gpt-4o"},
    {"name": "anthropic", "model": "claude-3-7-sonnet-20250219"},
    {"name": "deepinfra", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    {"name": "deepinfra", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
    {"name": "deepinfra", "model": "meta-llama/Meta-Llama-3.1-405B-Instruct"},
    {"name": "deepinfra", "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"},
    {"name": "deepinfra", "model": "Qwen/Qwen3-32B"}
]

# BabyAI levels (from the paper)
# Map difficulty levels 1,4,7,10,13,16,19 to the actual levels
all_levels = [
    "GoToObj", "GoToRedBallGrey", "GoToRedBall", "GoToLocal", "PutNextLocal", 
    "PickupLoc", "GoToObjMaze", "GoTo", "Pickup", "UnblockPickup", 
    "Open", "Synth", "SynthLoc", 
    "GoToSeq", "SynthSeq", "BossLevel"
]



'''

# PLAN #

all_levels = ['CustomBabyAI-GoToRedBall-Small-4Dists-v0',
 'CustomBabyAI-GoToRedBall-Small-5Dists-v0',
 'CustomBabyAI-GoToRedBall-Small-6Dists-v0',
 'CustomBabyAI-GoToRedBall-Small-7Dists-v0',
 'CustomBabyAI-GoToRedBall-Medium-20Dists-v0',
 'CustomBabyAI-GoToRedBall-Medium-40Dists-v0',
 'CustomBabyAI-GoToRedBall-Medium-50Dists-v0',
 'CustomBabyAI-GoToRedBall-Medium-60Dists-v0',
 'CustomBabyAI-GoToRedBall-Large-60Dists-v0',
 'CustomBabyAI-GoToRedBall-Large-80Dists-v0',
 'CustomBabyAI-GoToRedBall-Large-100Dists-v0',
 'CustomBabyAI-GoToRedBall-Large-120Dists-v0',
 'CustomBabyAI-GoToRedBall-Ultra-120Dists-v0',
 'CustomBabyAI-GoToRedBall-Ultra-140Dists-v0',
 'CustomBabyAI-GoToRedBall-Ultra-160Dists-v0',
 'CustomBabyAI-GoToRedBall-Ultra-180Dists-v0']

selected_levels = all_levels 
tasks = ["plan"]

'''

# Select the 7 difficulty levels mentioned in the todo
# difficulty_levels = [1, 2, 3, 4, 5, 6, 7, 8] # [9, 10, 11, 12, 13, 14, 15, 16]
selected_levels = all_levels # [all_levels[i-1] for i in difficulty_levels]

tasks = ["predict"]
prompters = ["tot"]
# prompters = ["zero_shot", "few_shot", "cot", "tot"]
formatters = ["structured"]

# Define 5 different seeds for the 5 instances
seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# Function to get a clean model identifier for filenames
def clean_for_filename(s):
    """Clean a string to be safe for use in filenames"""
    # Replace slashes, spaces and other problematic characters
    s = s.replace('/', '_').replace('\\', '_').replace(' ', '_')
    s = s.replace(':', '_').replace('*', '_').replace('?', '_')
    s = s.replace('"', '_').replace('<', '_').replace('>', '_')
    s = s.replace('|', '_').replace(';', '_').replace('=', '_')
    return s

# Generate configurations
config_count = 0
for llm, level, task, prompter, formatter, instance in product(
    llms, selected_levels, tasks, prompters, formatters, range(20)
):
    # Get the seed for this instance
    seed = seeds[instance]
    
    # Clean the model name for use in filenames
    clean_model = clean_for_filename(llm["model"])
    
    # Create configuration dictionary
    config = {
        "env": {
            "env_name": f"BabyAI-{level}-v0",
            "seed": seed  # Using different seeds for each instance
        },
        "inputs": {
            "formatter": formatter,
            "processor": "omniscient_babyai_bot",
            "prompter": prompter
        },
        "eval": {
            "task": task,
            "evaluator": task
        },
        "llm": {
            "name": llm["name"],
            "model": llm["model"],
            "temperature": 1.0,
            "max_tokens": 8192, # 1024,
            "system_prompt": ""
        },
        "output": {
            "path": f"results/{llm['name']}_{clean_model}_{level}_{task}_{prompter}_{formatter}_seed{seed}.json",
            "save_prompts": True,
            "save_raw_llm_output": True,
            "log_every": 10
        }
    }
    
    # Create a filename that includes both provider name and full model name
    filename = f"{llm['name']}_{clean_model}_{level}_{task}_{prompter}_{formatter}_seed{seed}.yaml"
    filepath = os.path.join(output_dir, filename)
    
    # Write to file
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    config_count += 1
    
    # Print progress every 100 configs
    if config_count % 100 == 0:
        print(f"Generated {config_count} configuration files so far...")

print(f"Generated total of {config_count} configuration files in the '{output_dir}' directory.")

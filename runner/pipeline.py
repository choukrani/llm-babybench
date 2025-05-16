# runner/pipeline.py

import os
import json
from pathlib import Path
from typing import Dict, Any

from runner.env_loader import load_envs
from runner.components import (
    get_formatter,
    get_processor,
    get_prompter,
    get_llm,
    get_evaluator,
)

def run_pipeline(config: Dict[str, Any]):

    envs = load_envs(config)
    # print(envs)

    # Get the base output path from the config
    base_output_path = Path(config.get("output", {}).get("path", "results/results.json"))
    
    # Create a unique filename by appending a timestamp
    import time
    timestamp = int(time.time())
    unique_output_path = base_output_path.with_name(f"{base_output_path.stem}_{timestamp}{base_output_path.suffix}")
    
    # Ensure the directory exists
    unique_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    formatter = get_formatter(config["inputs"]["formatter"])
    processor = get_processor(config["inputs"]["processor"])
    prompter = get_prompter(config["inputs"]["prompter"])
    evaluator = get_evaluator(config["eval"]["evaluator"])
    llm = get_llm(**config["llm"])

    env_name = config["env"]["env_name"]
    seed = config["env"]["seed"]
    task = config["eval"]["task"]

    results = []

    for idx, env in enumerate(envs):
        
        env_description = formatter.format(env)
        process = processor.process(env_name, seed, task)

        if task == 'predict':
            prompt = prompter.prompt(env_description, task, process)
        else:
            prompt = prompter.prompt(env_description, task)

        from llms.utils import parser
        all_llm_output = llm.generate(prompt) 
        predicted_output = parser(all_llm_output, task)
        print(all_llm_output)

        if task == 'predict':
            eval_result = evaluator.evaluate(env_name, seed, str_action_seq=process, predicted_output=predicted_output)

        elif task == 'plan':
            eval_result = evaluator.evaluate(env_name, seed, optimal_action_seq=process, llm_action_seq=predicted_output)
        
        elif task == 'decompose':
            eval_result = evaluator.evaluate(env, llm_output=predicted_output)


        result_entry = {
            "env_name": config["env"]["env_name"],
            "env_seed": config["env"]["seed"],
            "task": config["eval"]["task"],
            "formatter": config["inputs"]["formatter"],
            "prompter": config["inputs"]["prompter"],
            "prompt": prompt,
            "all_llm_output": all_llm_output,
            "parsed_llm_output": predicted_output,
            "eval_result": eval_result,
        }

        results.append(result_entry)

    # Save results to a unique file
    try:
        with open(unique_output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation complete. Results saved to: {unique_output_path}")
    except Exception as e:
        print(f"Error saving results to {unique_output_path}: {str(e)}")
        # Try saving to a fallback location
        fallback_path = Path(f"results/fallback_{os.getpid()}_{timestamp}.json")
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fallback_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to fallback location: {fallback_path}")

    return unique_output_path  # Return the path where results were saved

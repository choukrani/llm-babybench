# evaluators/predict.py

import re
from typing import Any, Union, Tuple, Dict 

import gymnasium as gym

from evaluators.base import AbstractEvaluator
from evaluators.metrics import manhattan_distance
from evaluators.utils import str_action_seq_to_int, parse_state_prediction


class PredictEvaluator(AbstractEvaluator):
    """
    Evaluator for BabyAI-Predict:
    (Env Description, Start State, Action Sequence) -> Final State
    """
    def evaluate(self, env_name: str, seed: int, str_action_seq: str, predicted_output: str, **kwargs) -> Dict[str, Any]:
        """
        Based on the environment (and the agent's start state),
        the function performs the action sequence in the env and compares LLM's predicted final state with ground truth.
        
        Args:
            env_name: The BabyAI MiniGrid environment name.
            seed: Random seed for the environment.
            str_action_seq: the action sequence as a string.
            predicted_output: LLM's predicted final state as a string.
            
        Returns:
            Dict of evaluation metrics.
        """
        # Create and initialize environment
        local_env = gym.make(env_name, tile_size=32, render_mode='rgb_array')
        local_env.reset(seed=seed)
        
        # Parse predicted final state
        predicted_state = parse_state_prediction(predicted_output)

        if not predicted_state:
            raise ValueError("predicted_state cannot be None or empty.")
        
        # Convert string action sequence to integers
        # print("LLM String Action Sequence: ", str_action_seq)
        int_action_seq = str_action_seq_to_int(str_action_seq)
        # print("Integer Action Sequence: ", int_action_seq)
        
        # Execute actions to get actual final state
        for action in int_action_seq:
            obs, reward, terminated, truncated, info = local_env.step(action)
            if terminated or truncated:
                break
        
        # Get final state after executing all actions
        final_state = (tuple(int(x) for x in local_env.unwrapped.agent_pos), int(local_env.unwrapped.agent_dir))
        # print("Actual Final State: ", final_state)
        
        # Clean up
        local_env.close()
        
        # Calculate metrics
        return {
            "success": predicted_state == final_state,
            "accuracy": float(predicted_state == final_state),
            # "manhattan_distance": manhattan_distance(predicted_state, final_state) if predicted_state else float('inf')
            "manhattan_distance": manhattan_distance(predicted_state, final_state)
        }

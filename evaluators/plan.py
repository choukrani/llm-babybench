# evaluators/plan.py

from typing import Any, Tuple, Dict

import gymnasium as gym

from evaluators.base import AbstractEvaluator
from evaluators.metrics import manhattan_distance
from evaluators.utils import str_action_seq_to_int

class PlanEvaluator(AbstractEvaluator):
    """
    Evaluator for BabyAI-Plan:
    (Env Description, Start State, Subgoal) -> Action Sequence
    """
    def evaluate(self, env_name: str, seed: int, optimal_action_seq: str, llm_action_seq: str, **kwargs) -> Dict[str, Any]:
        """
        Runs LLM's predicted action sequence in the env,
        compares LLM's response with optimal baseline,
        and checks for subgoal success.
        
        Args:
            env_name: The BabyAI MiniGrid environment name.
            seed: The BabyAI MiniGrid environment seed.
            optimal_action_seq: BabyAIBot optimal action sequence.
            llm_action_seq: The LLM predicted action sequence as a string.
            
        Returns:
            Dict of evaluation metrics.
        """
        # Convert action sequences from string to integers
        bot_actions = str_action_seq_to_int(optimal_action_seq)
        llm_actions = str_action_seq_to_int(llm_action_seq)
        
        # Execute optimal actions to get baseline performance
        env_bot = gym.make(env_name, tile_size=32, render_mode='rgb_array')
        env_bot.reset(seed=seed)
        bot_reward = 0
        bot_terminated = False
        bot_truncated = False
        bot_final_state = None
        
        for action in bot_actions:
            obs, reward, terminated, truncated, info = env_bot.step(action)
            bot_reward += reward
            if terminated or truncated:
                bot_terminated = terminated
                bot_truncated = truncated
                break

        # Get final state of Bot
        bot_final_state = (tuple(int(x) for x in env_bot.unwrapped.agent_pos), int(env_bot.unwrapped.agent_dir))
        
        # Execute LLM actions to get LLM performance
        env_llm = gym.make(env_name, tile_size=32, render_mode='rgb_array')
        env_llm.reset(seed=seed)
        llm_reward = 0
        llm_terminated = False
        llm_truncated = False
        
        for action in llm_actions:
            obs, reward, terminated, truncated, info = env_llm.step(action)
            llm_reward += reward
            if terminated or truncated:
                llm_terminated = terminated
                llm_truncated = truncated
                break
        
        # Get final state of LLM agent
        llm_final_state = (tuple(int(x) for x in env_llm.unwrapped.agent_pos), int(env_llm.unwrapped.agent_dir))
        
        # Find the red ball position for distance calculation
        red_ball_pos = None
        for i in range(env_llm.unwrapped.grid.width):
            for j in range(env_llm.unwrapped.grid.height):
                cell = env_llm.unwrapped.grid.get(i, j)
                if cell is not None and cell.type == 'ball' and cell.color == 'red':
                    red_ball_pos = (i, j)
                    break
            if red_ball_pos is not None:
                break

        red_ball_pos = (red_ball_pos, 0)

        # Calculate Manhattan distance between the two finale states
        llm_ball_distance = manhattan_distance(llm_final_state, red_ball_pos) - 1
        
        # Clean up
        env_bot.close()
        env_llm.close()
        
        return {
            "CR": 1 if llm_terminated else 0,  # Completion rate based on termination
            "number_of_bot_actions": len(bot_actions),
            "number_of_llm_actions": len(llm_actions),
            "llm_efficiency": len(bot_actions)/len(llm_actions) if len(llm_actions) > 0 and llm_terminated else 0,
            "bot_reward": bot_reward, # from minigrid
            "llm_reward": llm_reward, # from minigrid
            "llm_ball_distance": llm_ball_distance, # d(llm_final_state, ball)
            "llm_terminated": llm_terminated,  # Did LLM reach goal state
            "bot_terminated": bot_terminated   # Did optimal bot reach goal state
        }
    
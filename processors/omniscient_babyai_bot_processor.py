# preocessors/omniscient_babyai_bot_processor.py

from typing import Any, Optional, Tuple, List

import gymnasium as gym

from custom.custom_babyai_bots.omniscient_fixed_plan_baby_ai_bot import BabyAIBot

from processors.base import EnvProcessor
from processors.utils import int_action_seq_to_str


class OmniscientBabyAIBotProcessor(EnvProcessor):

    def process(
        self, 
        env_name: str, 
        seed: int, 
        task: Optional[str] = None, 
        max_steps: int = 1000,
        ) -> Any:

        # Create a local copy of the processed environment
        local_env = gym.make(env_name, render_mode='rgb_array', tile_size=32)

        # Reset with the same seed
        obs, info = local_env.reset(seed=seed)
    
        # Create the bot
        try:
            bot = BabyAIBot(local_env, [])
            bot.provide_initial_subgoals()
        except Exception as e:
            raise ValueError(f"Failed to create BabyAIBot. Is this a BabyAI-compatible environment? Error: {e}")
    
        # Initialize statistics
        stats = {
            'steps': 0,
            'total_reward': 0,
            'success': False,
            'gave_up': False
        }

        # Initialize action sequence
        action_seq = []

        # Solve the environment
        done = False
        while not done and stats['steps'] < max_steps:
                
            # Get the next action from the bot
            # action = bot.replan(obs)
            action = bot.replan()
            
            # Store action
            action_seq.append(int(action))
            
            # Execute the action
            obs, reward, terminated, truncated, info = local_env.step(action)
            
            # Update statistics
            stats['steps'] += 1
            stats['total_reward'] += reward
            
            # Check if the episode is done
            done = terminated or truncated
        
        # Set final statistics
        stats['success'] = terminated  # True if goal reached, False if max steps exceeded
        stats['gave_up'] = stats['steps'] >= max_steps and not done

        # Clean up
        local_env.close()
        
        # Return results based on task
        if task:
            return int_action_seq_to_str(action_seq)
        
        else:
            return stats

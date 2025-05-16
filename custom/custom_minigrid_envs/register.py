# custom/custom_minigrid_envs/register.py

from gymnasium.envs.registration import register
from custom_minigrid_envs.goto import CustomGoToRedBallEnv

def register_envs():
    """Register all custom environments with Gymnasium"""
    
    # Small single room environments
    for num_dists in [4, 5, 6, 7]:
        register(
            id=f'CustomBabyAI-GoToRedBall-Small-{num_dists}Dists-v0',
            entry_point='custom_minigrid_envs.goto:CustomGoToRedBallEnv',
            kwargs={
                'room_size': 8,
                'num_rows': 1,
                'num_cols': 1,
                'num_dists': num_dists
            }
        )

    # Medium single room environments
    for num_dists in [20, 40, 50, 60]:
        register(
            id=f'CustomBabyAI-GoToRedBall-Medium-{num_dists}Dists-v0',
            entry_point='custom_minigrid_envs.goto:CustomGoToRedBallEnv',
            kwargs={
                'room_size': 16,
                'num_rows': 1,
                'num_cols': 1,
                'num_dists': num_dists
            }
        )

    # Large single room environments
    for num_dists in [60, 80, 100, 120]:
        register(
            id=f'CustomBabyAI-GoToRedBall-Large-{num_dists}Dists-v0',
            entry_point='custom_minigrid_envs.goto:CustomGoToRedBallEnv',
            kwargs={
                'room_size': 24,
                'num_rows': 1,
                'num_cols': 1,
                'num_dists': num_dists
            }
        )

    # Very Large (Ultra) single room environments
    for num_dists in [120, 140, 160, 180]:
        register(
            id=f'CustomBabyAI-GoToRedBall-Ultra-{num_dists}Dists-v0',
            entry_point='custom_minigrid_envs.goto:CustomGoToRedBallEnv',
            kwargs={
                'room_size': 32,
                'num_rows': 1,
                'num_cols': 1,
                'num_dists': num_dists
            }
        )

    print("Custom environments registered successfully!")

# custom/custom_minigrid_envs/__init__.py

import os
import sys

# Add the custom folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the project root to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# Import and expose the registration function for easy access
from custom_minigrid_envs.register import register_envs

# Import environment classes if you want to expose them directly
from custom_minigrid_envs.goto import CustomGoToRedBallEnv

# Automatically register environments when the package is imported
register_envs()

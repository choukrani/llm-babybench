# formatters/json_formatter.py

import json
from typing import Dict, Any

import gymnasium as gym
from gymnasium import Env

from minigrid.core.world_object import Door, Key, Ball, Box

from formatters.base import EnvFormatter
from formatters.utils import agent_direction


class JSONFormatter(EnvFormatter):

    def format(self, env: Env) -> str:
        # Unpack environment attributes & ensure all extracted values are Python-native types
        num_rows = int(env.unwrapped.num_rows)
        num_cols = int(env.unwrapped.num_cols)
        room_size = int(env.unwrapped.room_size)
        agent_view_size = int(env.unwrapped.agent_view_size)
        agent_init_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
        agent_front_pos = tuple(int(x) for x in env.unwrapped.front_pos)
        agent_dir = int(env.unwrapped.agent_dir)
        grid_height = int(env.unwrapped.grid.height)
        grid_width = int(env.unwrapped.grid.width)
        mission = str(env.unwrapped.mission)

        # Context paragraph
        context = (
            "An agent is in a grid world consisting of one or more rooms. "
            "All rooms in the same grid world are squares of identical size and are organized in a square grid layout. "
            "Rooms are separated by walls and might contain objects such as keys, balls, and boxes of different colors. "
            "Some walls, connecting two adjacent rooms, have doors. Some doors are unlocked, whereas others need to be unlocked with keys of the same color. "
            "The agent can perform 6 actions: left (turn left), right (turn right), forward (move forward), pickup (pickup an object), drop (drop an object), and toggle (open/close a door or a box). "
            "Only the forward action changes the agent's position in the grid world. Turning left or right changes the agent's orientation only but not the position. "
            "The agent cannot move into a cell that is already occupied by an object, even if the object is one it is trying to interact with. "
            "Using a coordinate system where the (0, 0) position is the top-left corner of the grid world, necessarily corresponding to a wall, "
            "the coordinates follow the format (x, y), with x denoting the horizontal position in the grid and y denoting the vertical position in the grid,\n\n"
            "These are the specifics regarding this environment: \n\n"
        )

        # Environment config
        config: Dict[str, Any] = {
            "num_rooms": [num_rows, num_cols] if num_rows > 1 or num_cols > 1 else 1,
            "room_size_incl_walls": [room_size, room_size],
            "room_size_excl_walls": [room_size - 2, room_size - 2],
            "grid_size": [grid_height, grid_width],
            "agent_initial_pos": list(agent_init_pos),
            "agent_front_pos": list(agent_front_pos),
            "agent_direction": {"index": agent_dir, "name": agent_direction(agent_dir)},
            # "agent_view_size": [agent_view_size, agent_view_size],
            "objects": [],
            "mission": mission,
        }

        # Extract objects in the environment
        for idx, obj in enumerate(env.unwrapped.grid.grid):
            if isinstance(obj, (Door, Key, Ball, Box)):
                j = int(idx // grid_width)
                i = int(idx % grid_width)
                pos = [i, j]
                obj_data = {"type": obj.type, "color": obj.color, "position": pos}
                if isinstance(obj, Door):
                    obj_data["locked"] = bool(obj.is_locked)
                config["objects"].append(obj_data)

        # Wrap in final JSON structure
        output = {"context": context, "config": config}

        return str(json.dumps(output, indent=2))
    
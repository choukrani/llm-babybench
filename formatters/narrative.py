# formatters/narrative.py

import gymnasium as gym
from gymnasium import Env

from minigrid.core.world_object import Door, Key, Ball, Box

from formatters.base import EnvFormatter
from formatters.utils import agent_direction


class NarrativeFormatter(EnvFormatter):

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


        context = ""

        # handling the 1 room case

        if num_rows == 1 and num_cols == 1:
            # single room description
            context += f"An agent is in a grid world made of 1 room, this room of size {room_size}x{room_size}, including the surrounding walls, meaning that effectively, the room is of size {room_size - 2}x{room_size - 2}. "
            context += f"The total grid size is thus {grid_height}x{grid_width}. The room might contain objects such as keys, balls, and boxes of different colors. "
        else:
            # multi-room description
            context += f"An agent is in a grid world made of {num_rows}x{num_cols} rooms, each of size {room_size}x{room_size}, including the surrounding walls, meaning that effectively, each room is of size {room_size - 2}x{room_size - 2}. "
            context += f"The total grid size is thus {grid_height}x{grid_width}. Rooms are separated by walls and might contain objects such as keys, balls, and boxes of different colors. "
            context += f"Some walls, connecting two adjacent rooms, have doors. Some doors are unlocked, whereas others need to be unlocked with keys of the same color. "

        # shared description for both cases
        context += f"The agent can perform 6 actions: left (turn left), right (turn right), forward (move forward), pickup (pickup an object), drop (drop an object), and toggle (open/close a door or a box). "
        context += f"Only the forward action changes the agent's position in the grid world. Turning left or right changes the agent's orientation only but not the position. "
        context += f"The agent cannot move into a cell that is already occupied by an object, even if the object is one it is trying to interact with. "
        context += f"Using a coordinate system where the (0, 0) position is the top-left corner of the grid world, necessarily corresponding to a wall, "
        context += f"the coordinates follow the format (x, y), with x denoting the horizontal position in the grid and y denoting the vertical position in the grid, "
        context += f"and the agent is initially placed at {agent_init_pos}, and is facing, {agent_direction(agent_dir)}, the {agent_front_pos} position. "

        objects_in_env = []

        for t, obj in enumerate(env.unwrapped.grid.grid):
            if (
                isinstance(obj, Door)
                or isinstance(obj, Key)
                or isinstance(obj, Ball)
                or isinstance(obj, Box)
            ):
                j = t // env.unwrapped.grid.width
                i = t % env.unwrapped.grid.width
                objects_in_env.append(
                    (
                        obj.color,
                        obj.type,
                        (i, j),
                        obj.is_locked if isinstance(obj, Door) else None,
                    )
                )

        if len(objects_in_env) == 0:
            context += "There are no objects in the environment. "
        else:
            context += "There is "
            for t, (color, obj_type, pos, locked) in enumerate(objects_in_env):
                if t == len(objects_in_env) - 1 and len(objects_in_env) > 1:
                    context += f"and {'a' if obj_type != 'door' else ('a locked' if locked else 'an unlocked')} {color} {obj_type} at position {pos}. "
                else:
                    context += f"{'a' if obj_type != 'door' else ('a locked' if locked else 'an unlocked')} {color} {obj_type} at position {pos}"
                    if len(objects_in_env) > 1:
                        context += ", "
                    else:
                        context += ". "
        
        context += f"The agent\'s mission is '{env.unwrapped.mission}'. "

        return context
    
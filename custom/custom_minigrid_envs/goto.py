# custom/custom_minigrid_envs/goto.py

from minigrid.envs.babyai.goto import GoToRedBallGrey

class CustomGoToRedBallEnv(GoToRedBallGrey):
    """
    Custom version of GoToRedBallGrey with configurable grid dimensions
    
    Parameters:
    -----------
    room_size : int
        Size of each room (default: 8)
    num_rows : int
        Number of rooms in vertical direction (default: 1)
    num_cols : int
        Number of rooms in horizontal direction (default: 1)
    num_dists : int
        Number of distractor objects (default: 7)
    """
    def __init__(self, room_size=8, num_rows=1, num_cols=1, num_dists=7, **kwargs):
        # Call RoomGridLevel constructor directly with our parameters
        # instead of GoToRedBallGrey which sets fixed dimensions
        from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
        RoomGridLevel.__init__(
            self,
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            **kwargs
        )
        # Store the number of distractors
        self.num_dists = num_dists

    def gen_mission(self):
        """Generate a mission (place objects and define the goal)"""
        self.place_agent()
        
        # Generate a random room position for multi-room environments
        if self.num_rows > 1 or self.num_cols > 1:
            room_i = self.np_random.integers(0, self.num_cols)
            room_j = self.np_random.integers(0, self.num_rows)
        else:
            # Default to room (0,0) for single-room environments
            room_i, room_j = 0, 0
            
        # Place the red ball in the selected room
        obj, pos = self.add_object(room_i, room_j, "ball", "red")

        # Store the ball's position for easy access
        self.ball_room = (room_i, room_j)
        self.ball_pos = pos  # This is the absolute grid position
        
        # Add distractors
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        for dist in dists:
            dist.color = "grey"
            
        # Make sure no unblocking is required
        self.check_objs_reachable()
        
        # Create the instruction object
        from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
        
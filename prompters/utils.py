# /prompters/utils.py

from enum import Enum
from typing import Dict

# Enum to track task types internally
class Task(str, Enum):
    PREDICT = "predict"
    PLAN = "plan"
    DECOMPOSE = "decompose"

# Template dictionary 
TEMPLATES: Dict[str, Dict[Task, str]] = {

    "zero_shot": {
        Task.PREDICT: (
            "You are an intelligent agent. You are given the following environment description:\n{description}\n\n"
            "Predict the final state of the agent after executing the following action sequence:\n{str_action_seq}\n\n"
            "At the end of your response, write the agent's final state in the following way '((x, y), dir)'. "
            "For dir (agent facing): chose 0 for east, 1 for south, 2 for west, 3 for north. "
            "Begin this section with the phrase: 'The agent's final state is: '. "
            "Please do not write a full stop mark at the end! "
        ),
        Task.PLAN: (
            "You are an intelligent agent. You are given the following environment description:\n{description}\n\n"
            "So, your goal is to reach the red ball, and you need to determine the sequence of actions to get there. "
            "At the end of your response, write the action sequence in the following way action1, action2, action3 "
            "The names of possible actions are again: left, right, forward, pickup, drop, toggle."
            "Begin this section with the phrase: 'The LLM's action sequence is: '. "
            "Example: The LLM's action sequence is: forward, forward, left"
            "Please do not write a full stop mark at the end! "
        ),
        Task.DECOMPOSE: (
            "You are an intelligent agent. Given the environment description, decompose the task into a sequence of meaningful subgoals that you pick from:\n\n"
            "- OpenSubgoal: If the agent is next to a closed door, this subgoal will open it. \n\n"
            "- CloseSubgoal: If the agent is next to an open door, this subgoal will close it. \n\n"
            "- DropSubgoal: If the agent is carrying an object, this subgoal drops it. \n\n"
            "- PickupSubgoal: If the agent is not carrying any object and is next to an object, this subgoal picks it up. \n\n"
            "- GoNextToSubgoal, (x,y): If there is a clear, without any blocker, path between the agent and the cell of coordinates (x,y), this subgoal makes the agent go next to this cell. \n\n"
            "Environment Description:\n{description}\n\n"
            "Output will be parsed by a strict program. Strictly respect the format given below or parsing will fail.\n"
            "At the end of your answer, the last part of your answer will contain the sequence of subgoals presented using the following format:\n\n"
            "<START>\n(Subgoal_1)\n(Subgoal_2)\n... (Subgoal_n)\n<END>\n\n"
            "Where (Subgoal_i) can either be (OpenSubgoal), (CloseSubgoal),(DropSubgoal),(PickupSubgoal) or (GoNextToSubgoal, (x,y)).\n\n"
            "The list of subgoals should be in the chronological order, which means that the agent will follow Subgoal_1 then Subgoal_2 ..."
        ),
    },
    "few_shot": {
        Task.PREDICT: (
            "You are an intelligent agent. You are given the following environment description:\n{description}\n\n"
            "Predict the final state of the agent after executing the following action sequence:\n{str_action_seq}\n\n"
            "Here is a successful example for your reference:" # "Here are some successful examples for your reference:"
            "\n{FEW_SHOT_EXAMPLES[Task.PREDICT][0]}\n\n"
            # "{FEW_SHOT_EXAMPLES[Task.PREDICT][1]}\n\n"
            # "{FEW_SHOT_EXAMPLES[Task.PREDICT][2]}\n\n"
            "At the end of your response, write the agent's final state in the following way '((x, y), dir)'. "
            "For dir (agent facing): chose 0 for east, 1 for south, 2 for west, 3 for north. "
            "Begin this section with the phrase: 'The agent's final state is: '. "
            "Please do not write a full stop mark at the end! "
        ),
        Task.PLAN: (
            "You are an intelligent agent. You are given the following environment description:\n{description}\n\n"
            "So, your goal is to reach the red ball, and you need to determine the sequence of actions to get there. "
            "Here is a successful example for your reference:" # "Here are some successful examples for your reference:"
            "\n{FEW_SHOT_EXAMPLES[Task.PLAN][0]}\n\n"
            # "{FEW_SHOT_EXAMPLES[Task.PLAN][1]}\n\n"
            # "{FEW_SHOT_EXAMPLES[Task.PLAN][2]}\n\n"
            "At the end of your response, write the action sequence in the following way action1, action2, action3 "
            "The names of possible actions are again: left, right, forward, pickup, drop, toggle."
            "Begin this section with the phrase: 'The LLM's action sequence is: '. "
            "Example: The LLM's action sequence is: forward, forward, left"
            "Please do not write a full stop mark at the end! "
        ),
        Task.DECOMPOSE: (
            "You are an intelligent agent. Given the environment description bellow, decompose the task into a sequence of meaningful subgoals that you pick from:\n\n"
            "- OpenSubgoal: If the agent is next to a closed door, this subgoal will open it. \n\n"
            "- CloseSubgoal: If the agent is next to a open door, this subgoal will close it. \n\n"
            "- DropSubgoal: If the agent is carrying an object, this subgoal drops it. \n\n"
            "- PickupSubgoal: If the agent is not carrying any object and is next to an object, this subgoal picks it up. \n\n"
            "- GoNextToSubgoal, (x,y): If there is a clear, without any blocker, path between the agent and the cell of coordinates (x,y), this subgoal makes the agent go next to this cell. \n\n"
            "Environment Description:\n{description}\n\n"
            "Output will be parsed by a strict program. Strictly respect the format given below or parsing will fail.\n"
            "At the end of your answer, the last part of your answer will contain the sequence of subgoals presented using the following format:\n\n"
            "<START>\n(Subgoal_1)\n(Subgoal_2)\n... (Subgoal_n)\n<END>\n\n"
            "Where (Subgoal_i) can either be (OpenSubgoal), (CloseSubgoal),(DropSubgoal),(PickupSubgoal) or (GoNextToSubgoal, (x,y)).\n\n"
            
            "The list of subgoals should be in the chronological order, which means that the agent will follow Subgoal_1 then Subgoal_2 ..." 
            
            "Here is an example to help you: \n"

            "\n{FEW_SHOT_EXAMPLES[Task.DECOMPOSE][0]}\n\n" 

            "Example Input: An agent is in a grid world. The agent starts at position (3, 4) facing north. "
            "The mission is 'open the yellow door'. There is a yellow door at position (12, 7).\n"
            "Example output:\n"
            "The sequence of subgoals is:\nGoNextToSubgoal, (12,7)\nOpenSubgoal\n"
        ),
    },
    "cot": {
        Task.PREDICT: (
            "You are an intelligent agent. You are given the following environment description:\n{description}\n\n"
            "Let's think step-by-step to predict the agent's final state after executing the following action sequence:\n{str_action_seq}\n\n"

            "Step 1. Note the agent's starting position and direction\n"
            "Step 2. Process each action in sequence:\n"
            "   - For turns: Update the agent's direction\n"
            "   - For forward: Move in the current direction if not blocked\n"
            "   - For toggle/pickup/drop: Execute without changing position\n"
            "Step 3. Track the agent's position after each step\n"
            "Step 4. Determine the final position (x,y) and direction\n\n"

            "Here is a small example of a reasoning process:\n"
            "- Starting at (3, 4) facing north (dir=3)\n"
            "- Action 'right': Now facing east (dir=0), still at (3, 4)\n"
            "- Action 'forward': Move east to (4, 4), still facing east (dir=0)\n"
            "- Action 'pickup': Still at (4, 4) facing east (dir=0), but now holding an object\n\n"

            "At the end of your response, write the agent's final state in the following way '((x, y), dir)'. "
            "For dir (agent facing): chose 0 for east, 1 for south, 2 for west, 3 for north. "
            "Begin this section with the phrase: 'The agent's final state is: '. "
            "Please do not write a full stop mark at the end!"
        ),
        Task.PLAN: (
            "You are an intelligent agent. You are given the following environment description:\n{description}\n\n"
            "So, your goal is to reach the red ball, and you need to determine the sequence of actions to get there. "

            "Let's plan step-by-step to solve this MiniGrid environment task.\n\n"

            "Step 1. Identify the mission goal\n"
            "Step 2. Map out the environment layout\n"
            "Step 3. Break down the mission into actions\n"
            "Step 4. For each action:\n"
            "   - Determine optimal path to relevant places\n"
            "   - Plan the necessary rotations\n"
            "Step 5. Verify the plan actually works\n\n"

            "Here is a small example for your reference:\n"
            "- Goal: Go to the red ball at (5, 3)\n"
            "- Agent starts at (2, 4) facing west \n"
            "- Need to turn right twice to face east: 'right, right' (or 'left, left')\n"
            "- Move forward 3 steps to reach the ball: 'forward, forward, forward'\n"
            "- Need to turn left once to face the ball (face north): 'left'\n"
            "- Complete sequence: 'right, right, forward, forward, forward, left'\n\n"

            "At the end of your response, write the action sequence in the following way action1, action2, action3 "
            "The names of possible actions are again: left, right, forward, pickup, drop, toggle."
            "Begin this section with the phrase: 'The LLM's action sequence is: '. "
            "Example: The LLM's action sequence is: forward, forward, left"
            "Please do not write a full stop mark at the end! "
        ),
        Task.DECOMPOSE: (
            "You are an intelligent agent. Given the environment description bellow, let's plan step-by-step to decompose the task into a sequence of meaningful subgoals.\n\n"
            
            "Environment Description:\n{description}\n\n"

            "Step 1. Identify the mission goal\n"
            "- What is the agent ultimately required to achieve?\n\n"

            "Step 2. Identify relevant objects and obstacles\n"
            "- What doors, keys, objects, or blockers are present?\n"
            "- Which objects need to be picked up, moved, or interacted with?\n\n"

            "Step 3. Determine the necessary interactions\n"
            "- Which doors must be opened or closed?\n"
            "- Are there objects that must be picked up or dropped?\n\n"

            "Step 4. Plan the movement strategy\n"
            "- For each object or location the agent must interact with, decide where it must go next to.\n"
            "- Use (GoNextToSubgoal, (x,y)) if there's a clear path to that location.\n\n"

            "Step 5. Convert each step into a subgoal from the allowed list\n"
            "The available subgoals are:\n"
            "- (OpenSubgoal): If the agent is next to a closed door, this subgoal opens it.\n"
            "- (CloseSubgoal): If the agent is next to an open door, this subgoal closes it.\n"
            "- (DropSubgoal): If the agent is carrying an object, this subgoal drops it.\n"
            "- (PickupSubgoal): If the agent is not carrying any object and is next to an object, this subgoal picks it up.\n"
            "- (GoNextToSubgoal, (x, y)): If there is a clear, unobstructed path to cell (x, y), this subgoal moves the agent next to it.\n\n"

            "Step 6. Sequence the subgoals in logical order\n"
            "- Subgoals should be in the order the agent will execute them.\n"
            "- Make sure movement (GoNextToSubgoal) comes *before* interaction subgoals (like Pickup or Open).\n\n"
            
            "Here is a small example of step-by-step planning:\n"
            "-Goal: Go to the red ball at coordinates (x_b, y_b)"
            "There is a door between the agent and the red ball at coordinate (x,y). So first Subgoal is : GoNextToSubgoal, (x,y) \n"
            "To continue its travel to the red ball, the agents has to open the door. Next subgoal: OpenSubgoal\n"
            "Path to the ball is clear. Last Subgoal: GoNextToSubgoal, (x_b,y_b)\n\n"
            
            "Output will be parsed by a strict program. Strictly respect the format given below or parsing will fail.\n"
            "At the end of your answer, the last part of your answer will contain the sequence of subgoals presented using the following format:\n\n"
            "<START>\n(Subgoal_1)\n(Subgoal_2)\n... (Subgoal_n)\n<END>\n\n"
            "Where (Subgoal_i) can either be (OpenSubgoal), (CloseSubgoal),(DropSubgoal),(PickupSubgoal) or (GoNextToSubgoal, (x,y)).\n\n"
            "The list of subgoals should be in the chronological order, which means that the agent will follow Subgoal_1 then Subgoal_2 ..."
        ),
    },
    "tot": {
        Task.PREDICT: (
            "You are an intelligent agent. You are given the following environment description:\n{description}\n\n"
            "Let's predict, using tree-of-thought reasoning, the agent's final state after executing the following action sequence:\n{str_action_seq}\n\n"

            "Initial state: [Identify agent's starting position and direction]\n\n"
            "Action sequence analysis:\n"
            "* For each action, consider multiple possibilities:\n"
            "  - What direction will the agent face?\n"
            "  - What position will the agent be in?\n"
            "  - Are there obstacles that would prevent movement?\n"
            "  - What state changes occur in the environment?\n\n"
            
            "Example of tree-of-thought reasoning:\n"
            "Starting state: (3, 4) facing west\n"
            "Action 'left':\n"
            "- Option 1: Turn to face north\n"
            "- Option 2: Turn to face south [Correct since left from west is south]\n" 
            
            "New state: (3, 4) facing south\n\n"
            "Action 'forward':\n"
            "- Option 1: Move to (3, 5) [Correct since moving south increases y]\n"
            "- Option 2: Can't move forward if there's an obstacle at (3, 5) [Check environment]\n"
            "- Selected option: Move to (3, 5), no obstacle present\n\n"
            
            "When the process is done, make sure to trace the complete execution path and converge on the final state.\n\n"

            "At the end of your response, write the agent's final state in the following way '((x, y), dir)'. "
            "For dir (agent facing): chose 0 for east, 1 for south, 2 for west, 3 for north. "
            "Begin this section with the phrase: 'The agent's final state is: '. "
            "Please do not write a full stop mark at the end! "
        ),
        Task.PLAN: (

            "You are an intelligent agent. You are given the following environment description:\n{description}\n\n"
            "So, your goal is to reach the red ball, and you need to determine the sequence of actions to get there using tree-of-thought planning. "
            
            "Mission analysis:\n"
            "* What is the primary goal? [Identify]\n"
            "* What are the potential obstacles? [List]\n\n"

            "Example of tree-of-thought planning:\n"
            "Goal: Go to the red ball at (5, 2)\n"
            "Current position: (3, 4) facing west\n\n"
            
            "Path Option 1: Turn right twice and go east, then left and go north \n"
            "- Actions: right, right, forward, forward, left, forward \n"
            "- Analysis: Requires 6 actions, but might encounter obstacle at (4, 4) \n\n"
            
            "Path Option 2: Turn right and go north, then east \n"
            "- Actions: right, forward, forward, right, forward \n"
            "- Analysis: Requires 5 actions, more efficient if path is clear \n\n"
            
            "Selected approach: Path Option 2 (shorter and avoids potential obstacle)\n\n"
            
            "Final plan assembly:\n"
            "* Combine chosen approaches for each trajectory\n"
            "* Eliminate redundant actions\n"
            "* Verify completeness\n\n"

            "At the end of your response, write the action sequence in the following way action1, action2, action3 "
            "The names of possible actions are again: left, right, forward, pickup, drop, toggle."
            "Begin this section with the phrase: 'The LLM's action sequence is: '. "
            "Example: The LLM's action sequence is: forward, forward, left"
            "Please do not write a full stop mark at the end! "
        ),
        Task.DECOMPOSE: (

            "You are an intelligent agent. Let's plan step-by-step and with a tree-of-thoughts approach to decompose the grid task bellow into a sequence of meaningful subgoals.\n\n"
            "Environment Description:\n{description}\n\n"

            "Step 1. Identify the mission goal\n"
            "- What is the agent ultimately required to achieve?\n\n"

            "Step 2. Identify relevant objects and obstacles\n"
            "- What doors, keys, objects, or blockers are present?\n"
            "- Which objects need to be picked up, moved, or interacted with?\n\n"

            "Step 3. Determine the necessary interactions\n"
            "- Which doors must be opened or closed?\n"
            "- Are there objects that must be picked up or dropped?\n\n"

            "Step 4. Plan at least two options of movement\n"
            "- For each object or location the agent must interact with, decide where it must go next to.\n"
            "- Use (GoNextToSubgoal, (x,y)) if there's a clear path to that location.\n\n"

            "Step 5 Choose the best option.\n"
            "- Evaluate the options based on usefulness, efficiency, and feasibility.\n"
            "- Select the better subgoal to commit to.\n"

            "Step 6. Convert each step into a subgoal from the allowed list\n"
            "The available subgoals are:\n"
            "- (OpenSubgoal): If the agent is next to a closed door, this subgoal opens it.\n"
            "- (CloseSubgoal): If the agent is next to an open door, this subgoal closes it.\n"
            "- (DropSubgoal): If the agent is carrying an object, this subgoal drops it.\n"
            "- (PickupSubgoal): If the agent is not carrying any object and is next to an object, this subgoal picks it up.\n"
            "- (GoNextToSubgoal, (x, y)): If there is a clear, unobstructed path to cell (x, y), this subgoal moves the agent next to it.\n\n"

            "Step 7. Sequence the subgoals in logical order\n"
            "- Subgoals should be in the order the agent will execute them.\n"
            "- Make sure movement (GoNextToSubgoal) comes *before* interaction subgoals (like Pickup or Open).\n\n"
            
            "Example of step-by-step tree-of-thought planning:\n"
            "- Goal: Go to the red ball at coordinates (x_b, y_b)\n\n"
            "Step 1:\n"
            "Candidate 1: (GoNextToSubgoal, (x, y))  # Move next to the door blocking the path to the ball\n"
            "Candidate 2: (GoNextToSubgoal, (x_b, y_b))  # Try to go directly to the ball\n"
            "Evaluation:\n"
            "- Candidate 2 is not feasible because the door at (x, y) blocks the path.\n"
            "- Candidate 1 prepares the agent to interact with the door.\n"
            "Chosen subgoal: (GoNextToSubgoal, (x, y))\n\n"
            "Step 2:\n"
            "Candidate 1: (OpenSubgoal)  # Open the door to access the path\n"
            "Candidate 2: (PickupSubgoal)  # Try picking something up now\n"
            "Evaluation:\n"
            "- Candidate 2 is invalid; there is nothing to pick up yet.\n"
            "- Candidate 1 is necessary to unlock access to the ball.\n"
            "Chosen subgoal: (OpenSubgoal)\n\n"
            "Step 3:\n"
            "Candidate 1: (GoNextToSubgoal, (x_b, y_b))  # Proceed directly to the red ball\n"
            "Candidate 2: (CloseSubgoal)  # Close the door behind\n"
            "Evaluation:\n"
            "- Candidate 2 is unnecessary at this point.\n"
            "- Candidate 1 directly progresses toward the goal.\n"
            "Chosen subgoal: (GoNextToSubgoal, (x_b, y_b))\n\n"

            "Output will be parsed by a strict program. Strictly respect the format given below or parsing will fail.\n"
            "At the end of your answer, the last part of your answer will contain the sequence of subgoals presented using the following format:\n\n"
            "<START>\n(Subgoal_1)\n(Subgoal_2)\n... (Subgoal_n)\n<END>\n\n"
            "Where (Subgoal_i) can either be (OpenSubgoal), (CloseSubgoal),(DropSubgoal),(PickupSubgoal) or (GoNextToSubgoal, (x,y)).\n\n"        ),
    },
}


# Example banks with more detailed examples for few-shot learning
FEW_SHOT_EXAMPLES = {

    Task.PREDICT: [

        (
            "Environment description:\n\n"
            "An agent is in a grid world made of 1 room, this room of size 8x8, including the surrounding walls, "
            "meaning that effectively, the room is of size 6x6. The total grid size is thus 8x8. "
            "The room might contain objects such as keys, balls, and boxes of different colors. "
            "The agent can perform 6 actions: left (turn left), right (turn right), forward (move forward), pickup (pickup an object), drop (drop an object), and toggle (open/close a door or a box). "
            "Only the forward action changes the agent's position in the grid world. Turning left or right changes the agent's orientation only but not the position. "
            "The agent cannot move into a cell that is already occupied by an object, even if the object is one it is trying to interact with. "
            "Using a coordinate system where the (0, 0) position is the top-left corner of the grid world, necessarily corresponding to a wall, "
            "the coordinates follow the format (x, y), with x denoting the horizontal position in the grid and y denoting the vertical position in the grid, "
            "and the agent is initially placed at (6, 4), and is facing, south, the (6, 5) position. "
            "There is a red key at position (5, 1). The agent's mission is 'go to the red key'. "

            "\n\nThe agent is now supposed to follow the following action sequence: right, forward, forward, forward, left, forward\n\n"

            "Starting at (6, 4), turning right will not change the position, but it will change the direction. The agent was facing south, and by turning right, it will be facing west. "
            "The agent is now at position (6, 4) facing west, by moving forward three times in a row in this direction, it will be at position (3, 4) since moving west decreases the x coordinate. "
            "Now, at position (3, 4), moving left will not change the position, but it will change the direction. The agent was facing west, and by turning left, it will be facing south. "
            "The agent is now at position (3, 4) facing south, by moving forward in this direction, it will be at position (3, 5) since moving south increases the y coordinate. "

            "\n\nThe agent is finally at position (3, 5) facing south. "
            "South corresponds to 2. "
            
            "\n\nThe agent's final state is: ((3, 5), 2)"
        ), 
        
        (
            "example2"
            "example2"
        )
        
        ],

    Task.PLAN: [

        (
            "Environment description:\n\n"

            "An agent is in a grid world made of 1 room, this room of size 8x8, including the surrounding walls, "
            "meaning that effectively, the room is of size 6x6. The total grid size is thus 8x8. "
            "The room might contain objects such as keys, balls, and boxes of different colors. "
            "The agent can perform 6 actions: left (turn left), right (turn right), forward (move forward), pickup (pickup an object), drop (drop an object), and toggle (open/close a door or a box). "
            "Only the forward action changes the agent's position in the grid world. Turning left or right changes the agent's orientation only but not the position. "
            "The agent cannot move into a cell that is already occupied by an object, even if the object is one it is trying to interact with. "
            "Using a coordinate system where the (0, 0) position is the top-left corner of the grid world, necessarily corresponding to a wall, "
            "the coordinates follow the format (x, y), with x denoting the horizontal position in the grid and y denoting the vertical position in the grid, "

            "and the agent is initially placed at (5, 3), and is facing, south, the (5, 4) position. "
            "There is a grey ball at position (5, 2), a red ball at position (5, 1), a grey ball at position (1, 2), a grey ball at position (3, 3), "
            "a grey box at position (6, 6), a grey ball at position (1, 1), and a grey ball at position (1, 6). "
            
            
            "\n\nThe agent's mission is 'go to the red ball'. "
            

            "Starting at (5, 3), turning left will not change the position, but it will change the direction. The agent was facing south, and by turning left, it will be facing east. "
            "The agent is now at position (5, 3) facing east, by moving forward in this direction, it will be at position (6, 3) (which is free) since moving east increases the x coordinate. "
            "Now, at position (6, 3), moving left will not change the position, but it will change the direction. The agent was facing east, and by turning left, it will be facing north. "
            "The agent is now at position (6, 3) facing north, by moving forward twice in a row in this direction, it will be at position (6, 1) (which is free) since moving north decreases the y coordinate. "
            "Now, at position (6, 1), moving left will not change the position, but it will change the direction. The agent was facing north, and by turning left, it will be facing west, directly the red ball in position (5, 1). "

            "\n\nThe LLM's action sequence is: left, forward, left, forward, forward, left"
            
        ), 
        
        (
            "example2"
            "example2"
        )
        
        ],

    Task.DECOMPOSE: [

        (
            "example1"
            "example1"
        ), 
        
        (
            "example2"
            "example2"
        )
        
        ],
}

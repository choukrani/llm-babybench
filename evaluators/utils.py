# evaluators/utils.py

import re

from typing import Any, Tuple, List, Union

from minigrid.core.actions import Actions

from custom.custom_babyai_bots.omniscient_fixed_plan_baby_ai_bot import (
    BabyAIBot,
    Subgoal,
    ExploreSubgoal,
    GoNextToSubgoal,
    PickupSubgoal,
    OpenSubgoal,
    CloseSubgoal,
    DropSubgoal,
)


def str_action_seq_to_int(str_action_seq: str) -> List[int]:  

    action_names = [a.strip().lower() for a in str_action_seq.split(",")]

    try:
        return [Actions[a].value for a in action_names if a in Actions.__members__]
    
    except KeyError as e:
        raise ValueError(f"Invalid action in input: {e}")


def int_action_seq_to_str(int_action_seq: List[int]) -> str:

    str_action_seq = ""

    for i, int_action in enumerate(int_action_seq):
        str_action_seq += ["left", "right", "forward", "pickup", "drop", "toggle", "done"][int_action]
        if i != len(int_action_seq) - 1:
            str_action_seq += ", "

    return str_action_seq


def parse_state_prediction(prediction_str: str) -> Union[Tuple[Tuple[int, int], int], None]:
    """
    Parse the predicted state from the LLM's output string.
    Expected format: ((x, y), direction)
    
    Args:
        prediction_str: String containing the predicted state
        
    Returns:
        Parsed state tuple or None if parsing fails
    """
    try:
        # Use regex to find the pattern ((x, y), dir)
        pattern = r'\(\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*(-?\d+)\s*\)'
        match = re.search(pattern, prediction_str)
        
        if match:
            x, y, direction = map(int, match.groups())
            return ((x, y), direction)
        
        # Fallback: just extract any three integers
        numbers = re.findall(r'-?\d+', prediction_str)
        if len(numbers) >= 3:
            x, y, direction = map(int, numbers[:3])
            return ((x, y), direction)
        
        return None
    except Exception as e:
        print(f"Error parsing prediction: {e}")
        return None
    

def parse_str_subgoals(strings: List[str]) -> List[Tuple[str, Any]]:
    """
    Input: list of strings like ['(GoNextToSubgoal, (3,5))', '(PickupSubgoal)']
    Output: list of tuples like [('GoNextToSubgoal', (3,5)), ('PickupSubgoal')]
    """
    def parse_one_subgoal(input_str: str):
        s = input_str.strip()

        # Remove optional outer parentheses
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1].strip()

        # Match with inner tuple: GoNextToSubgoal, (3,5)
        match_with_tuple = re.fullmatch(r'([^,]+?)\s*,\s*\(\s*([^\(\)]+?)\s*\)', s)
        # Match without tuple: GoNextToSubgoal
        match_without_tuple = re.fullmatch(r'([^,]+)', s)

        if match_with_tuple:
            first = match_with_tuple.group(1).strip()
            second = tuple(map(int, map(str.strip, match_with_tuple.group(2).split(','))))
            return (first, second)
        elif match_without_tuple:
            first = match_without_tuple.group(1).strip()
            return (first, None)
        else:
            raise ValueError(f"Input string format not recognized: {input_str!r}")
    
    result = []
    for x in strings: result.append(parse_one_subgoal(x))
    return result


def instantiate_subgoals(bot, subgoals: List[Tuple]) -> List[Subgoal]:
    """
    Instantiate the subgoals produced by the LLM.
    subgoals is a list of tuples like ('GoNextToSubgoal',(3,5)). They are ordered in the chronological order. If no position is given, it is None.
    """
    new_subgoals = []
    for sg in subgoals:
        sg_type, sg_pos = sg
        if sg_type == "ExploreSubgoal":
            new_subgoal = ExploreSubgoal(bot, datum=sg_pos)
        elif sg_type == "GoNextToSubgoal":
            new_subgoal = GoNextToSubgoal(bot, datum=sg_pos)
        elif sg_type == "PickupSubgoal":
            new_subgoal = PickupSubgoal(bot, datum=sg_pos)
        elif sg_type == "OpenSubgoal":
            new_subgoal = OpenSubgoal(bot, datum=sg_pos)
        elif sg_type == "CloseSubgoal":
            new_subgoal = CloseSubgoal(bot, datum=sg_pos)
        elif sg_type == "DropSubgoal":
            new_subgoal = DropSubgoal(bot, datum=sg_pos)
        else:
            raise ValueError(f"Unknown subgoal type: {sg_type}")
        new_subgoals.append(new_subgoal)
    return new_subgoals

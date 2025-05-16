# processors/utils.py

from typing import List

def int_action_seq_to_str(int_action_seq: List[int]) -> str:

    str_action_seq = ""

    for i, int_action in enumerate(int_action_seq):
        str_action_seq += ["left", "right", "forward", "pickup", "drop", "toggle", "done"][int_action]
        if i != len(int_action_seq) - 1:
            str_action_seq += ", "

    return str_action_seq
    

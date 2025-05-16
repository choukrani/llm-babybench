# evaluators/utils.py

from typing import Tuple

def manhattan_distance(pred_state: Tuple[Tuple[int, int], int], true_state: Tuple[Tuple[int, int], int]) -> int:
    ((x1, y1), dir1) = pred_state
    ((x2, y2), dir2) = true_state
    distance = abs(x1 - x2) + abs(y1 - y2)
    return distance

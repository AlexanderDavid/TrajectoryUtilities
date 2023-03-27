from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, List, Optional, Callable

if TYPE_CHECKING:
    from ..dataset import Agent

import numpy as np

def trajectory_regularity(ego: Agent) -> float:
    ideal_dist = np.linalg.norm(ego.goal - ego.start)
    actual_dist = np.sum(
        [
            np.linalg.norm(ego.positions[i + 1].pos - ego.positions[i].pos)
            for i in range(len(ego.positions) - 1)
        ]
    )

    return ideal_dist / actual_dist

def speed(
    ego: Agent, func: Optional[Callable[np.array, float]] = None
) -> Union[float, List[float]]:
    speeds = np.array([np.linalg.norm(x.vel) for x in ego.positions])

    if func is not None:
        return func(speeds)

    return speeds


def travel_time(ego: Agent) -> float:
    return ego.positions[-1].time - ego.positions[0].time


def straight_line_displacement(
    ego: Agent, func: Optional[Callable[np.array, float]] = None
) -> Union[float, List[float]]:
    start = ego.positions[0].pos
    end = ego.positions[-1].pos

    distances = np.array(
        [
            np.abs(
                np.linalg.norm(np.cross(end - start, start - pos.pos))
                / np.linalg.norm(end - start)
            )
            for pos in ego.positions
        ]
    )

    if func is not None:
        return np.mean(distances)

    return distances


def straight_line_time_displacement(ego: Agent) -> List[float]:
    start_p = ego.positions[0].pos
    end_p = ego.positions[-1].pos

    start_t = ego.positions[0].time
    end_t = ego.positions[-1].time

    displacements = []
    for pos in ego.positions:
        fractional_time = (pos.time - start_t) / (end_t - start_t)
        straight_line_distance = start_p + ((end_p - start_p) * fractional_time)
        displacements.append(pos.pos - straight_line_distance)

    return displacements

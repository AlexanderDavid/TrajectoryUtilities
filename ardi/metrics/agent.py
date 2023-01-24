from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, List, Optional, Callable

if TYPE_CHECKING:
    from ..dataset import Agent

import numpy as np


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

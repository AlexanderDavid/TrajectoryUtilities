from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, List, Optional, Callable, Tuple
from enum import Enum
from .util import resample_trajectory

if TYPE_CHECKING:
    from ..dataset import Agent

import numpy as np


def curvature(ego: Agent):
    def angle(u1: np.array, u2: np.array) -> float:
        result = np.clip(
            np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)), -1, 1
        )
        return np.arccos(result)

    curvatures = []

    x, y = resample_trajectory(ego, len(ego.positions))
    points = np.vstack((x, y)).T

    for i in range(1, len(x) - 1):
        u1 = points[i] - points[i - 1]
        u2 = points[i + 1] - points[i]

        deltaTheta = angle(u1, u2)
        curvatures.append(
            np.power(deltaTheta / ((np.linalg.norm(u1) + np.linalg.norm(u2)) / 2), 2)
        )

    return curvatures


def curvature_fdm(ego: Agent):
    """Calculates the curvature of a trajectory using the finite difference method

    Args:
        ego (Agent): Agent to calculate curvature of trajectory
    """

    x, y = resample_trajectory(ego, len(ego.positions))

    dx = np.gradient(x)
    dy = np.gradient(y)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    return np.abs(ddx * dy - dx * ddy) / np.power(dx**2 + dy**2, 3 / 2)


def energy_efficiency(
    ego: Agent, b: float = 2.25, c: float = 1, sum: bool = True
) -> float:
    es = []
    ps = []
    ees = []
    for pos in ego.positions:
        e = b + c * np.linalg.norm(pos.vel) ** 2
        p = np.dot(pos.vel, (ego.goal - pos.pos) / np.linalg.norm(ego.goal - pos.pos))

        es.append(e)
        ps.append(p)
        ees.append(p / e)

    if sum:
        return np.sum(ees)

    return es, ps, ees


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
        displacements.append(np.linalg.norm(pos.pos - straight_line_distance))

    return displacements


def distance_overhead(ego: Agent) -> float:
    ideal_distance = np.linalg.norm(ego.positions[-1].pos - ego.positions[0].pos)

    actual_distance = 0
    for i in range(len(ego.positions) - 1):
        actual_distance += np.linalg.norm(
            ego.positions[i].pos - ego.positions[i + 1].pos
        )

    return actual_distance / ideal_distance

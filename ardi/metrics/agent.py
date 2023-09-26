from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, List, Optional, Callable
from enum import Enum
from .util import resample_trajectory, resample_trajectory_min_distance

if TYPE_CHECKING:
    from ..dataset import Agent

import numpy as np

"""
TODO: 
[ ] Have the resampling functions spit out a new List[Position] rather than two numpy arrays
"""

class CurvatureMethod(Enum):
    SQUARE = 0
    FINITE_DIFFERENCE = 1

class ResampleMethod(Enum):
    MIN_DISTANCE = 0
    NUM_POINTS = 1

def curvature(ego: Agent, curve_method: CurvatureMethod=CurvatureMethod.SQUARE, 
              resample_method: ResampleMethod=ResampleMethod.MIN_DISTANCE,
              resample_arg: Union[float, int]=0.1) -> List[float]:
    """Calculate the curvature of a path using either the square curvature or the finite difference
    method. The resampling can also be changed to either sample such that no two points are closer than
    some x or such that there are x points in the trajectory

    Args:
        ego (Agent): Trajectory to calculate after
        curve_method (CurvatureMethod, optional): Method to calculate curvature. Defaults to CurvatureMethod.SQUARE.
        resample_method (ResampleMethod, optional): Method to resample trajectory with. Defaults to ResampleMethod.MIN_DISTANCE.
        resample_arg (Union[float, int], optional): Arg to pass to resampling method. Defaults to 0.1.

    Raises:
        NotImplementedError: If either the curvature or resampling method is invalid

    Returns:
        List[float]: Curvature over the entire resampled trajectory. Usually just passed
        to a reduction function like np.mean or np.sum
    """
    
    if resample_method == ResampleMethod.MIN_DISTANCE:
        x, y = resample_trajectory_min_distance(ego, resample_arg)
    elif resample_method == ResampleMethod.NUM_POINTS:
        x, y = resample_trajectory(ego, resample_arg)
    else:
        raise NotImplementedError("Resample Method invalid")
        
    if curve_method == CurvatureMethod.SQUARE:
        return __square_curvature(x, y)

    elif curve_method == CurvatureMethod.FINITE_DIFFERENCE:
        return __fdm_curvature(x, y)

    else:
        raise NotImplementedError("Curvature Method invalid")

def __square_curvature(xs: np.array, ys: np.array) -> List[float]:
    assert len(xs) == len(ys)

    def angle(u1: np.array, u2: np.array) -> float:
        result = np.clip(
            np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)), -1, 1
        )
        return np.arccos(result)

    curvatures = []

    for i in range(1, len(xs) - 1):
        u1 = xs[i] - xs[i - 1]
        u2 = ys[i + 1] - ys[i]

        deltaTheta = angle(u1, u2)
        curvatures.append(
            np.power(deltaTheta / ((np.linalg.norm(u1) + np.linalg.norm(u2)) / 2), 2)
        )

    return curvatures


def __fdm_curvature(xs: np.array, ys: np.array) -> List[float]:
    assert len(xs) == len(ys)
    dx = np.gradient(xs)
    dy = np.gradient(ys)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    return np.abs(ddx * dy - dx * ddy) / np.power(dx**2 + dy**2, 3 / 2)


def energy_efficiency(
    ego: Agent, b: float = 2.25, c: float = 1) -> List[float]:
    """Calculate the energy efficiency of an agent as defined in
    
    Godoy, J., Karamouzas, I., Guy, S.J., Gini, M., 2014. Anytime
    navigation with progressive hindsight optimization, in: 
    IEEE/RSJ Int. Conf. on Intelligent Robots and Systems.

    Args:
        ego (Agent): Trajectory to calculate energy efficiency over
        b (float, optional): Scaling factor for movement. Defaults to 2.25.
        c (float, optional): Scaling factor for thinking. Defaults to 1.

    Returns:
        List[float]: Energy effieciencies at each timestep
    """
    ees = []
    for pos in ego.positions:
        e = b + c * np.linalg.norm(pos.vel) ** 2
        p = np.dot(pos.vel, (ego.goal - pos.pos) / np.linalg.norm(ego.goal - pos.pos))

        ees.append(p / e)

    return ees


def trajectory_regularity(ego: Agent) -> float:
    """ Return the ratio of the ideal distance, or the distance as the crow
    flies, and the actual distance (the sum of the norms of differences in position)
    as a metric dealing with how much overhead there is in the path

    Args:
        ego (Agent): Agent to calculate over

    Returns:
        float: Distance overhead
    """
    ideal_dist = np.linalg.norm(ego.goal - ego.start)
    actual_dist = np.sum(
        [
            np.linalg.norm(ego.positions[i + 1].pos - ego.positions[i].pos)
            for i in range(len(ego.positions) - 1)
        ]
    )

    return ideal_dist / actual_dist


def speed(ego: Agent) -> List[float]:
    speeds = np.array([np.linalg.norm(x.vel) for x in ego.positions])
    return speeds


def travel_time(ego: Agent) -> float:
    return ego.positions[-1].time - ego.positions[0].time


def straight_line_displacement( ego: Agent) -> List[float]:
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

from __future__ import annotations
from math import sqrt
from typing import TYPE_CHECKING, Callable, List, Optional
from .util import get_over_trajectory

if TYPE_CHECKING:
    from ..dataset import Position, Agent

import numpy as np
from tslearn.metrics import dtw


def ratio_distance_violations(
    agent: Agent, other: Agent, alpha: float, beta: float
) -> int:
    distances = get_over_trajectory(
        agent, other, lambda x, y: np.linalg.norm(x.pos - y.pos)
    )
    close = 0
    too_close = 0

    for d in distances:
        if d >= beta and d <= alpha:
            close += 1
        elif d <= beta:
            too_close += 1

    if close == 0 and too_close == 0:
        return 0
    elif close == 0:
        return 1

    return too_close / close


def ratio_ttc_violations(agent: Agent, other: Agent, alpha: float) -> float:
    ttc_values = get_over_trajectory(
        agent, other, lambda x, y: ttc(x, y, agent.radius + other.radius)
    )
    violations = float(len([x for x in ttc_values if x <= alpha]))
    valids = float(len([x for x in ttc_values if x != float("inf")]))

    if valids == 0:
        return 0

    return violations / valids


def ttc(agent: Position, obstacle: Position, radius_sum: float) -> float:
    """Calculate the Time to Collision for two agents

    Args:
        agent (Position): Current agent configuration
        obstacle (Position): Current obstacle configuration

    Returns:
        float: Time till collision assuming two agents keep their
                current heading and velocity
    """
    w = agent.pos - obstacle.pos
    c = w.dot(w) - radius_sum

    if c < 0:
        return 0

    v = agent.vel - obstacle.vel
    a = v.dot(v)
    b = w.dot(v)

    if b > 0:
        return float("inf")

    discr = b * b - a * c
    if discr <= 0:
        return float("inf")

    tau = c / (-b + sqrt(discr))

    if tau < 0:
        return float("inf")

    return tau


def ttca(agent: Position, obstacle: Position) -> float:
    """Calculate the Time to Closest Approach as defined in eqution 3 from

        Zhang, Bingqing, et al. "From HRI to CRI: Crowd Robot Interaction—
        Understanding the Effect of Robots on Crowd Motion."
        International Journal of Social Robotics 14.3 (2022): 631-643.

    Args:
        agent (Position): Current agent configuration
        obstacle (Position): Current obstacle configuration

    Returns:
        float: Time till closest approach assuming two agents keep their
                current heading and velocity
    """
    v_o_a = obstacle.vel - agent.vel

    if not np.any(v_o_a):
        return 0

    p_o_a = obstacle.pos - agent.pos

    return -np.dot(p_o_a, v_o_a) / np.linalg.norm(v_o_a) ** 2


def mpds(
    ego: Agent,
    other: Agent,
    filter_fn: Optional[Callable[[Position, Position], bool]] = None,
) -> List[float]:
    mpds = []
    for pos in ego.positions:
        for pos_ in other.positions:
            if pos.time != pos_.time or (
                filter_fn is not None and filter_fn(pos, pos_)
            ):
                continue

            mpds.append(mpd(pos, pos_))

    return mpds


def mpd(agent: Position, obstacle: Position) -> float:
    """Calculate the projected minimum predicted distance between two agent's at a specified
    timestep.


    Args:
        agent (Position): Ego agent
        obstacle (Position): Other agent or obstacle

    Returns:
        float: Point-to-point minimum predicted distance
    """
    ttca_val = ttca(agent, obstacle)
    p_o_a = obstacle.pos - agent.pos

    if ttca_val < 0:
        return np.linalg.norm(p_o_a)

    v_o_a = obstacle.vel - agent.vel
    dca = np.linalg.norm(p_o_a + max(0, ttca_val) * v_o_a)

    return float(dca)

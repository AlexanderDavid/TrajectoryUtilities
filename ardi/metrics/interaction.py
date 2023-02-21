from __future__ import annotations
from math import sqrt
from typing import TYPE_CHECKING, Callable, List, Optional

from .util import get_over_trajectory, percentage_violations, sync_trajectories_in_time

if TYPE_CHECKING:
    from ..dataset import Position, Agent

import numpy as np
from tslearn.metrics import dtw


def count_distance_violations(agent: Agent, others: List[Agent], alpha: float) -> int:
    if len(others) == 0:
        return 0

    others = sync_trajectories_in_time(agent, others)

    distances = np.array(
        [
            get_over_trajectory(
                agent, other, lambda x, y: np.linalg.norm(x.pos - y.pos), float("inf")
            )
            for other in others
        ]
    )

    return len([x for x in np.min(distances, axis=0) if x <= alpha])


def ratio_distance_violations(
    agent: Agent, others: List[Agent], alpha: float, beta: float
) -> float:
    if len(others) == 0:
        return 0

    others = sync_trajectories_in_time(agent, others)

    distances = np.array(
        [
            get_over_trajectory(
                agent, other, lambda x, y: np.linalg.norm(x.pos - y.pos), float("inf")
            )
            for other in others
        ]
    )

    pv = percentage_violations(np.min(distances, axis=0), alpha, beta)

    return pv


def count_ttc_violations(agent: Agent, others: List[Agent], alpha: float) -> int:
    if len(others) == 0:
        return 0

    others = sync_trajectories_in_time(agent, others)

    ttc_values = np.array(
        [
            get_over_trajectory(
                agent,
                other,
                lambda x, y: ttc(x, y, agent.radius + other.radius),
                float("inf"),
            )
            for other in others
        ]
    )

    return len([x for x in np.min(ttc_values, axis=0) if x <= alpha])


def ratio_ttc_collision_free(agent: Agent, others: List[Agent], alpha: float) -> float:
    if len(others) == 0:
        return 0

    others = sync_trajectories_in_time(agent, others)

    ttc_values = np.array(
        [
            get_over_trajectory(
                agent,
                other,
                lambda x, y: ttc(x, y, agent.radius + other.radius),
                float("inf"),
            )
            for other in others
        ]
    )

    min_ttcs = np.min(ttc_values, axis=0)
    min_ttcs = min_ttcs[min_ttcs != np.array(None)]

    return len([x for x in min_ttcs if x >= alpha]) / len(min_ttcs)


def ratio_ttc_violations(
    agent: Agent, others: List[Agent], alpha: float, beta: float
) -> float:
    if len(others) == 0:
        return 0

    others = sync_trajectories_in_time(agent, others)

    ttc_values = np.array(
        [
            get_over_trajectory(
                agent,
                other,
                lambda x, y: ttc(x, y, agent.radius + other.radius),
                float("inf"),
            )
            for other in others
        ]
    )

    return percentage_violations(np.min(ttc_values, axis=0), alpha, beta)


def ttcs(agent: Agent, others: List[Agent]) -> List[float]:
    if len(others) == 0:
        return 0

    others = sync_trajectories_in_time(agent, others)

    mpd_values = np.array(
        [
            get_over_trajectory(
                agent,
                other,
                lambda x, y: ttc(x, y, agent.radius + other.radius),
                float("inf"),
            )
            for other in others
        ]
    )

    return np.min(mpd_values, axis=0)


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


def mpds(agent: Agent, others: List[Agent]) -> List[float]:
    if len(others) == 0:
        return 0

    others = sync_trajectories_in_time(agent, others)

    mpd_values = np.array(
        [
            get_over_trajectory(
                agent,
                other,
                lambda x, y: mpd(x, y),
                float("inf"),
            )
            for other in others
        ]
    )

    return np.min(mpd_values, axis=0)


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

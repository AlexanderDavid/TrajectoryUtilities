from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..dataset import Position, Agent

import numpy as np


def distance(agent: Agent, other: Agent) -> List[float]:
    distances = []
    for pos in agent.positions:
        for pos_ in other.positions:
            if pos.time != pos_.time:
                continue

            distances.append(np.linalg.norm(pos.pos - pos_.pos))

    return distances


def number_distance_violations(agent: Agent, other: Agent) -> int:
    pass


def ttc(agent: Position, obstacle: Position) -> float:
    """Calculate the Time to Collision for two agents as defined in the Powerlaw code from UMN:

    http://motion.cs.umn.edu/PowerLaw/

    Args:
        agent (Position): Current agent configuration
        obstacle (Position): Current obstacle configuration

    Returns:
        float: Time till collision assuming two agents keep their
                current heading and velocity
    """

    w = obstacle.pos - agent.pos
    v = agent.vel - obstacle.vel
    a = np.dot(v, v)
    b = np.dot(w, v)
    c = np.dot(w, w)
    discr = b * b - a * c

    if discr <= 0 or np.isclose(a, 0):
        return float("inf")

    discr = np.sqrt(discr)
    t1 = (b - discr) / a
    t2 = (b + discr) / a

    if t1 > t2:
        t1, t2 = t2, t1

    if t2 < 0:
        return float("-inf")

    if t1 < 0 and t2 > 0:
        return 0

    if t1 >= 0:
        return t1

    return float("inf")


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

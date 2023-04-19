from __future__ import annotations
from typing import TYPE_CHECKING, List, Callable, Any, Tuple
from warnings import warn
from copy import deepcopy

if TYPE_CHECKING:
    from ..dataset import Agent

import numpy as np
from scipy.interpolate import interp1d


def resample_trajectory(ego: Agent, num_samples: float) -> Tuple[np.array, np.array]:
    """Resample a trajectory for equally distant points

    Args:
        ego (Agent): Agen't to resample over
        num_samples (float): number of samples to use

    Returns:
        Tuple[np.array, np.array]: x and y components of trajectory
    """
    # Calculate the cumulative distance over the entire trajectory
    x = np.array([x.pos[0] for x in ego.positions])
    y = np.array([x.pos[1] for x in ego.positions])

    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cum_d = np.concatenate(([0], np.cumsum(d)))
    dist_resampled = np.linspace(0, cum_d[-1], num_samples)

    # Linearly interpolate over the distance
    f_x = interp1d(cum_d, x)
    f_y = interp1d(cum_d, y)

    return f_x(dist_resampled), f_y(dist_resampled)


def sync_trajectories_in_time(ego: Agent, others: List[Agent]) -> List[Agent]:
    """Syncronize the trajectories of a set of agents to the times in a reference agent. This means
       that t[0] for ego is t[0] for all agents in others and that no agent has a t[-1] > than ego t[-1].
       This does not mean that len(ego) == len(others) as the agents in others could have a shorter overall
       trajector

    Args:
        ego (Agent): Ego agent to sync to
        others (List[Agent]): Other agents to sync

    Returns:
        List[Agent]: Synced other agents
    """

    ego_start = ego.positions[0].time
    ego_end = ego.positions[-1].time

    others = deepcopy(others)

    for i in range(len(others)):
        others[i].positions = [
            x for x in others[i].positions if x.time >= ego_start and x.time <= ego_end
        ]
        if len(others[i].positions) <= len(ego.positions):
            ts = ego.positions[1].time - ego.positions[0].time
            last_t = ego.positions[-1].time * ts

            # Because of the dependencies between this utils class and the dataset class this is the hackiest way I have ever
            # instantiated a variable
            pos_cls = others[i].positions[0].__class__
            extend_len = len(ego.positions) - len(others[i].positions)
            others[i].positions += [
                pos_cls(
                    None, None, ego.positions[j + len(others[i].positions) - 1].time
                )
                for j in range(extend_len)
            ]

    return others


def percentage_violations(values: List[float], alpha: float, beta: float) -> float:
    """Calculate the percentage of values that fall below beta compared to those that fall
       below both alpha and beta.

    Args:
        values (List[float]): Values to compare
        alpha (float): Larger "bullseye" value
        beta (float): Smaller "bullseye" value

    Returns:
        float: Percentage of values that fall within beta compared to within alpha
    """
    assert alpha > beta
    values = values[values != np.array(None)]
    alphas = np.count_nonzero(values <= alpha)
    betas = np.count_nonzero(values <= beta)

    if alphas == 0:
        warn("No values found within the larger bullseye")
        return 0

    return betas / alphas


def get_over_trajectory(
    agent: Agent, other: Agent, fn: Callable[[Agent, Agent], Any], empty_value: Any
) -> List[Any]:
    """Iterate over a trajectory finding matching positions in time and return the results from
    some function on those positions

    Args:
        agent (Agent): Ego agent
        other (Agent): Other agent
        fn (Callable[[Agent, Agent], Any]): Function to call

    Returns:
        List[Any]: Result of the function over the trajectory
    """
    values = []
    for pos in agent.positions:
        for pos_ in other.positions:
            if pos.time != pos_.time:
                continue

            if pos_.pos is None:
                values.append(empty_value)
            else:
                values.append(fn(pos, pos_))

    return values

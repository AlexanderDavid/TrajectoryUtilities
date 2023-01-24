from __future__ import annotations
from typing import TYPE_CHECKING, List, Callable, Any

if TYPE_CHECKING:
    from ..dataset import Agent


def get_over_trajectory(
    agent: Agent, other: Agent, fn: Callable[[Agent, Agent], Any]
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

            values.append(fn(pos, pos_))

    return values

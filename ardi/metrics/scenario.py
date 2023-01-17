from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..dataset import Dataset


def last_arrival_time(dataset: Dataset, ignore: List = None) -> float:
    return max(
        [
            agent.positions[-1].time - agent.positions[0].time
            for agent in dataset.agents.values()
            if ignore is None or agent.idx not in ignore
        ]
    )

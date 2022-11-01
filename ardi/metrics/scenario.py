from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset import Dataset


def last_arrival_time(dataset: Dataset) -> float:
    return max(
        [
            agent.positions[-1].time - agent.positions[0].time
            for _, agent in dataset.agents.values()
        ]
    )

from ..dataset import Dataset


def last_arrival_time(dataset: Dataset) -> float:
    return max(
        [
            agent.positions[-1].time - agent.positions[0].time
            for _, agent in dataset.agents.values()
        ]
    )

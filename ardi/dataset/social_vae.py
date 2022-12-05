from pathlib import Path
from typing import Dict, List
from .abstract import Dataset, Agent, Position

import numpy as np


class SocialVAEDataset(Dataset):
    def _load(self, filename: Path, delim: str = " "):
        # Read the data in from the CSV
        data = np.loadtxt(str(filename), delimiter=delim, dtype=str)

        # Gather the agent numbers
        idxs = list(map(int, set(data[:, 1])))

        # Set up the base data structure of the dataset
        self._times = np.array(sorted(np.unique(data[:, 0]).astype(float)))
        self._timestep = self._times[1] - self._times[0]

        # Read through all of the trajectories for each individual agent
        for idx in idxs:
            agent_data = data[[int(datum[1]) == idx for datum in data]]

            goal = agent_data[0][7:9].astype(float)
            start = agent_data[0][2:4].astype(float)

            # Create the agent
            t = Agent(
                idx,
                agent_data[0][4],
                0.13,
                goal,
                start,
                pref_speed=0.5 if idx == -1 else 1.3,
            )

            # Get the positions and velocities into the positions array
            poss = agent_data[:, 2:4].astype(float)
            vels = np.vstack((np.zeros(2), (poss[:-1] - poss[1:])))
            times = agent_data[:, 0].astype(float)

            for pos, vel, time in zip(poss, vels, times):
                t.positions.append(Position(pos, -vel / self._timestep, time))

            self._agents[idx] = t

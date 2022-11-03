from typing import Dict, List
from .abstract import Dataset, Agent, Position

import numpy as np


class SocialVAEDataset(Dataset):
    def __init__(self, data_filename: str, actual_timestep: float=0.4):
        # Read the data in from the CSV
        self._data = np.loadtxt(data_filename, delimiter=" ", dtype=str)
        self._filename = data_filename

        # Gather the agent numbers
        self._idxs = list(map(int, set(self._data[:, 1])))

        # Set up the base data structure of the dataset
        self._agents: Dict[int, Agent] = {}
        self._times: List[float] = np.array(sorted(np.unique(self._data[:, 0]).astype(float))) * actual_timestep
        self._timestep = self._times[1] - self._times[0]

        # Read through all of the trajectories for each individual agent
        for idx in self._idxs:
            agent_data = self._data[[int(datum[1]) == idx for datum in self._data]]

            goal = agent_data[0][7:9].astype(float)
            start = agent_data[0][2:4].astype(float)

            # Create the agent
            t = Agent(idx, agent_data[0][4], 0.13, goal, start, pref_speed=0.5 if idx == -1 else 1.3)

            # Get the positions and velocities into the positions array
            poss = agent_data[:, 2:4].astype(float)
            vels = np.vstack((np.zeros(2), (poss[:-1] - poss[1:])))
            times = agent_data[:, 0].astype(float)

            for pos, vel, time in zip(poss, vels, times):
                t.positions.append(Position(pos, -vel / self._timestep, time * actual_timestep))

            self.agents[idx] = t

    @property
    def agents(self) -> Dict[int, Agent]:
        """Return a dictionary containing a map between agent index and
        that agent's information
        """
        return self._agents

    @property
    def timestep(self) -> float:
        """Return the difference between any two successive points in the dataset"""
        return self._timestep

    @property
    def times(self) -> float:
        return self._times

    def frameskip(self, skip: int) -> None:
        self._times = self._times[::skip]
        self._timestep *= skip

        for idx in self._agents:
            self._agents[idx].positions = [
                x for x in self._agents[idx].positions if x.time in self._times
            ]

    def trim_start(self, trim: int) -> None:
        self._times = self._times[trim:]

        for idx in self._agents:
            self._agents[idx].positions = [
                x for x in self._agents[idx].positions if x.time in self._times
            ]

    def trim_end(self, trim: int) -> None:
        self._times = self._times[:-trim]

        for idx in self._agents:
            self._agents[idx].positions = [
                x for x in self._agents[idx].positions if x.time in self._times
            ]

from dataclasses import dataclass, field
from collections.abc import KeysView
from pathlib import Path
from typing import List, Dict, Union, Tuple
import csv

import numpy as np
from matplotlib import pyplot as plt #type: ignore
import pandas as pd

from .abstract import Dataset, Agent, Position

COLORS_MAP = {
    -1: "#000000",
    0: "#ff0000",
    1: "#00ff00",
    2: "#0000ff"
}

RADIUS_MAP = {
    -1: 0.354 / 2,
    0: 0.4699 / 2,
    1: 0.4699 / 2,
    2: 0.5334 / 2,
    3: 0.4572 / 2,
    4: 0.4826 / 2,
    5: 0.4064 / 2,
    6: 0.4064 / 2,
    7: 0.4826 / 2,
    8: 0.4318 / 2
}

HEIGHT_MAP = {
    -1: 0.408,
    0: 1.8034,
    1: 1.8034,
    2: 1.8034,
    3: 1.905,
    4: 1.7526,
    5: 1.6256,
    6: 1.6256,
    7: 1.8796,
    8: 1.7272
}
    

class ZuckerDataset(Dataset):
    def __init__(self, data_filename: str, robot_idx: int=None, ttca_eps: float=0.05, start_speed: float=None):
        """Load a trajectory and optionally trim based on ttca and minimum speed

        Args:
            data_filename (str): Path to the dataset
            robot_idx (int, optional): Robot index that will be used to trim the end of the trajectory based on the TTCA. Defaults to None.
            start_speed (float, optional): Minimum starting speed that will be used to trim the beginning of the trajectory. Defaults to None.
        """
        # Try to get the id of the scene from the filename
        try:
            self._idx = int(Path(data_filename).stem)
        except Exception:
            print(f"dataset: cannot parse filename ({data_filename}) to find scenario index")
            self._idx = -1

        # Read the data in from the CSV
        self._data = np.loadtxt(data_filename, delimiter=",")
        self._filename = data_filename

        # Gather the agent numbers
        self._idxs = list(map(int, set(self._data[:, 0])))

        # Set up the base data structure of the dataset
        self._agents: Dict[int, Agent] = {}
        self._times: List[float] = np.unique(self._data[:, 1])
        self._timestep = self._times[1] - self._times[0]

        # Read through all of the trajectories for each individual agent
        for idx in self._idxs:
            agent_data = self._data[[datum[0] == idx for datum in self._data]]

            # Robot goal should always be the last position
            if idx == -1:
                goal = agent_data[-1][2:4]
            else:
                goal = agent_data[0][4:]

            # Create the agent
            t = Agent(idx, str(idx) if idx != -1 else "robot", RADIUS_MAP[idx], goal, COLORS_MAP[idx % 2 if idx != -1 else -1], HEIGHT_MAP[idx])
            
            # Get the positions and velocities into the positions array
            poss = agent_data[:, 2:4]
            vels = np.vstack((np.zeros(2), (poss[:-1] - poss[1:])))
            times = agent_data[:, 1]

            for pos, vel, time in zip(poss, vels, times):
                t.positions.append(Position(pos, -vel / self._timestep, time))

            self.agents[idx] = t

    @property
    def filename(self):
        return self._filename
    
    @property
    def id(self):
        return self._idx

    @property
    def agents(self):
        return self._agents

    @property
    def timestep(self):
        return self._timestep

    @property
    def times(self):
        return self._times

@dataclass
class ZuckerScenario:
    agent_type: str
    scene: str
    required: bool

    participants: List[int] = field(default_factory=list)

class ZuckerDatasetMap:
    def __init__(self, map_filename: str):
        """Load a map csv file

        Args:
            map_filename (str): Path to the map
        """
        self._map = {}
        rdr = csv.reader(open(map_filename), delimiter=",", quotechar='"')
        next(rdr) # Skip the header
        for row in rdr:
            if row[2][-1] == " ": row[2] = row[2][:-1]
            if row[3][-1] == " ": row[3] = row[3][:-1]
            self._map[int(row[1])] = ZuckerScenario(
                row[2],
                row[3],
                row[5] == "Y",
                list(map(int, row[4].replace(" ", "").split(",")))
            )

    def keys(self) -> KeysView:
        """Return all keys contianed in the map

        Returns:
            List[int]: List of numerical keys
        """
        return self._map.keys()
    
    def __getitem__(self, key: Union[int, Dataset]) -> ZuckerScenario:
        """Index the map and return meta information about that scenario

        Args:
            key (Union[int, Dataset]): Either an integer refering to the scenario index or a loaded dataset

        Raises:
            KeyError: If no Meta information is found in the loaded map

        Returns:
            Scenario: Meta information about the scneario
        """
        if type(key) is Dataset:
            if key.id != -1:
                return self._map[key.id]
            else:
                raise KeyError(key.id)

        return self._map[key] #type: ignore

    def get_meta_map(self, root: str) -> pd.DataFrame:
        data = []
        for fn in Path(root).glob("**/*csv"):
            if "map" in str(fn): continue
            if int(fn.stem) not in self.keys(): continue

            datum = self[int(fn.stem)]
            data.append({
                "filename": fn,
                "id": int(fn.stem),
                "agent_type": datum.agent_type,
                "controller": "human" if datum.agent_type.isnumeric() else datum.agent_type,
                "scene": datum.scene,
                "order": datum.participants
            })

        return pd.DataFrame(data)
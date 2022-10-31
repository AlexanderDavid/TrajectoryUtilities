from typing import Dict, List
from .abstract import Dataset, Agent

import numpy as np

class SocialVAEDataset(Dataset):
    def __init__(self, data_filename: str):
        # Read the data in from the CSV
        self._data = np.loadtxt(data_filename, delimiter=" ")
        self._filename = data_filename
        

    @property
    def agents(self) -> Dict[int, Agent]:
        """Return a dictionary containing a map between agent index and 
           that agent's information
        """

    @property
    def timestep(self) -> float:
        """Return the difference between any two successive points in the dataset
        """

    @property
    def times(self) -> List[float]:
        """Return a list of all times that are valid in the trajectory
        """

    def frameskip(self, skip: int) -> None:
        """In place frameskip the dataset.

        Args:
            skip (int): number of frames to skip
        """
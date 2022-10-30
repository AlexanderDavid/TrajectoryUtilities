from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from matplotlib import pyplot as plt
import numpy as np
from secrets import token_hex

@dataclass
class Position:
    pos: np.ndarray
    vel: np.ndarray
    time: float
    
    def __eq__(self, __o: object) -> bool:
        return np.allclose(__o.pos, self.pos) and np.allclose(__o.vel, self.vel) and __o.time == self.time

class Agent:
    def __init__(self, agent_idx: int, label: str, radius: float, goal: np.ndarray, color: Optional[str]=None, height: Optional[float]=None):
        self.idx = agent_idx
        self.label = label
        self.radius = radius
        self.goal = goal
        self.color = color if color else f"#{token_hex(3).upper()}"
        self.height = height

        self.positions: List[Position] = []

    def __repr__(self):
        return f"{self.idx=} w/ {len(self.positions)=}"

class Dataset(ABC):
    @property
    @abstractmethod
    def agents(self) -> Dict[int, Agent]:
        """Return a dictionary containing a map between agent index and 
           that agent's information
        """

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Return the difference between any two successive points in the dataset
        """

    @property
    @abstractmethod
    def times(self) -> List[float]:
        """Return a list of all times that are valid in the trajectory
        """

    @property
    def extents(self) -> Tuple[float, float, float, float]:
        """Return the minimum and maximum along x and y axis for all positions over all time.
        In format of (xmin xmax ymin ymax)
        """
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for idx in self.agents:
            xs = [x.pos[0] for x in self.agents[idx].positions]
            ys = [x.pos[1] for x in self.agents[idx].positions]

            min_x = min(min_x, min(xs))
            max_x = max(max_x, max(xs))

            min_y = min(min_y, min(ys))
            max_y = max(max_y, max(ys))

        return min_x, max_x, min_y, max_y

    def get_positions(self, time: float) -> Dict[int, Tuple[Position, Agent]]:
        """Return a dictionary mapping the index to a tuple of a specific position and the agent's information
        for all of all agents that are in the trajectory at a given time

        Args:
            time (float): Time to query the agents at, should be in self.times

        Returns:
            Dict[int, Tuple[Position, Agent]]: A dictionary mapping agent index to agent position and information at some given time
        """
        positions = {}

        for idx in self.agents:
            for pos in self.agents[idx].positions:
                if pos.time != time:
                    continue

                positions[idx] = (pos, self.agents[idx])

        return positions

    def align(self, agent_idx: int=None) -> None:
        """Rotate and Transform the scenario such that that a given agent starts at (0, 0) and has an initial 
        normalized goal vector of (1, 0)

        Args:
            agent_idx (int, optional): Agent to align, when None the minimum non-agent index is used. Defaults to None.
        """

        def rotate(point: np.ndarray, angle: float) -> np.ndarray:
            return np.array([
                np.cos(angle) * point[0] - np.sin(angle) * point[1],
                np.sin(angle) * point[0] + np.cos(angle) * point[1]
            ])

        if agent_idx is None:
            agent_idx = max(x for x in self.agents.keys() if x != -1)

        goal_vel = self.agents[agent_idx].goal - self.agents[agent_idx].positions[0].pos
        heading = -np.arctan2(goal_vel[1], goal_vel[0])

        transform = rotate(-self.agents[agent_idx].positions[0].pos, heading)

        
        for idx in self.agents:
            self.agents[idx].goal = rotate(self.agents[idx].goal, heading)
            self.agents[idx].goal += transform

            for i in range(len(self.agents[idx].positions)):
                self.agents[idx].positions[i].pos = rotate(self.agents[idx].positions[i].pos, heading)
                self.agents[idx].positions[i].pos += transform


    def display(self, ax: "AxesSubplot"=None) -> None:
        """Draw the agent's trajectory using matplotlib

        Args:
            ax (AxesSubplot, optional): Axes to plot to if you want to display in a subplot. If None plt is used. Defaults to None.
        """
        if ax == None:
            ax_ = plt
        else:
            ax_ = ax

        for idx in self.agents:
            # Start position
            ax_.scatter(
                [t.pos[0] for t in self.agents[idx].positions][0],
                [t.pos[1] for t in self.agents[idx].positions][0],
                c=self.agents[idx].color
            )

            # Full trajectory
            ax_.plot(
                [t.pos[0] for t in self.agents[idx].positions if t.time in self.times],
                [t.pos[1] for t in self.agents[idx].positions if t.time in self.times],
                label=str(idx) if idx != -1 else "Robot",
                c=self.agents[idx].color
            )

            # Goal position
            ax_.scatter(
                *self.agents[idx].goal,
                marker="x",
                s=40,
                c=self.agents[idx].color
            )
        
        if ax == None:
            plt.legend()
            plt.show()
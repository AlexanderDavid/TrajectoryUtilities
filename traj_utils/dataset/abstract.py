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
        return f"{self.agent.idx=} w/ {len(self.positions)=}"

class Dataset(ABC):
    @staticmethod
    def mpd(agent: Position, obstacle: Position) -> float:
        """ Calculate the projected minimum predicted distance between two agent's at a specified
        timestep.
        

        Args:
            agent (Position): Ego agent
            obstacle (Position): Other agent or obstacle

        Returns:
            float: Point-to-point minimum predicted distance 
        """
        ttca = Dataset.ttca(agent, obstacle)

        if ttca < 0:
            return float("inf")

        p_o_a = obstacle.pos - agent.pos
        v_o_a = obstacle.vel - agent.vel
        dca = np.linalg.norm(p_o_a + max(0, ttca) * v_o_a)

        return float(dca)

    @staticmethod
    def ttca(agent: Position, obstacle: Position) -> float:
        """Calculate the Time to Closest Approach as defined in eqution 3 from
        
         Zhang, Bingqing, et al. "From HRI to CRI: Crowd Robot Interaction—
            Understanding the Effect of Robots on Crowd Motion."
            International Journal of Social Robotics 14.3 (2022): 631-643.

        Args:
            agent (Position): Current agent configuration
            obstacle (Position): Current obstacle configuration

        Returns:
            float: Time till closest approach assuming two agents keep their 
                   current heading and velocity
        """
        v_o_a = obstacle.vel - agent.vel

        if not np.any(v_o_a):
            return 0
        
        p_o_a = obstacle.pos - agent.pos

        return -np.dot(p_o_a, v_o_a) / np.linalg.norm(v_o_a) ** 2

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
from abc import ABC, abstractmethod
from enum import Enum
from ..dataset import Position, Agent
from typing import List, Tuple, Dict

import numpy as np

class VelocityCalc(Enum):
    LAST_GROUND_TRUTH = 1
    AVERAGE_GROUND_TRUTH  = 2
    LAST_DISPLACEMENT = 3
    AVERAGE_DISPLACEMENT = 4

class Predictor(ABC):
    def __init__(self, frames: int, velocity_calc_method: VelocityCalc):
        self._frames = frames
        self._velocity_calc_method = velocity_calc_method
        
    @abstractmethod
    def predict(self, ego_history: Agent, neighbors_history: List[Agent]) -> List[List[Position]]:
        """Predict ego trajectory into the future

        Args:
            ego_history (Agent): ego agent with the history in the positions field
            neighbors_history (List[Agent]): neighbors with the history in the positions field

        Returns:
            List[Position]: prediction for the ego agent
        """

    def predict_ade_fde(self, ego_history: Agent, ego_truth: Agent, neighbors_history: List[Agent]) -> Tuple[float, float]:
        """Generate frames timesteps into the future with some prediction algorithm and then calculate
        the ADE and FDE of that prediction

        Args:
            ego_history (Agent): previous ego motion
            neighbor_history (List[Agent]): previous neighbor motion
            ego_truth (Agent): Actual ground truth motion the ego agent took

        Returns:
            Tuple[float, float]: Average displacement error, final displacement error
        """

        predicted = self.predict(ego_history, neighbors_history)

        difference = np.array([x.pos for x in predicted]) - np.array([x.pos for x in ego_truth.positions])
        distance = np.linalg.norm(difference, axis=1)

        return np.linalg.mean(distance), distance[-1]
    
    @staticmethod
    def calc_velocity(poss: List[Position], method: VelocityCalc=VelocityCalc.LAST_GROUND_TRUTH):
        if method == VelocityCalc.LAST_GROUND_TRUTH:
            return poss[-1].vel

        if method == VelocityCalc.AVERAGE_GROUND_TRUTH:
            return np.mean([p.vel for p in poss], axis=0)

        if method == VelocityCalc.LAST_DISPLACEMENT:
            return poss[-2].pos - poss[-1].pos

        if method == VelocityCalc.AVERAGE_DISPLACEMENT:
            positions = np.array([p.pos for p in poss])
            displacements = (positions[1:] - positions[:-1]) / (poss[1].time - poss[0].time)
            return np.mean(displacements, axis=0)
        
        raise ValueError("method must be a valid VelocityCalculation")

            
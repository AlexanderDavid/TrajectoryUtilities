from abc import ABC, abstractmethod
from enum import Enum
from ..dataset import Position, Agent
from typing import List, Tuple, Dict

import numpy as np

# This denotes some sub-path in a larger trajectory as well as agent
# meta information. Need to come up with a new type name because this
# doesn't really capture that it has meta information
SubTrajectory = Tuple[List[Position], Agent]

class VelocityCalc(Enum):
    LAST_GROUND_TRUTH = 1
    AVERAGE_GROUND_TRUTH  = 2
    LAST_DISPLACEMENT = 3
    AVERAGE_DISPLACEMENT = 4

class Predictor(ABC):
    @abstractmethod
    def predict(self, ego_history: SubTrajectory, neighbor_history: Dict[int, SubTrajectory], frames: int,
                velocity_calc_method: VelocityCalc) -> SubTrajectory:
        """Generate frames timesteps into the future with some prediction algorithm

        Args:
            ego_history (SubTrajectory): previous ego motion
            neighbor_history (Dict[int, SubTrajectory]): previous neighbor motion
            frames (int): number of frames to predict into the future
            velocity_calc_method (VelocityCalc): methods used to calculate velocity if that is needed
                                                 for the algorithm
        Returns:
            SubTrajectory: predicted future trajectory
        """

    def predict_ade_fde(self, ego_history: SubTrajectory, neighbor_history: Dict[int, SubTrajectory], ego_truth: SubTrajectory,
                        frames: int, velocity_calc_method: VelocityCalc) -> Tuple[float, float]:
        """Generate frames timesteps into the future with some prediction algorithm and then calculate
        the ADE and FDE of that prediction

        Args:
            ego_history (SubTrajectory): previous ego motion
            neighbor_history (Dict[int, SubTrajectory]): previous neighbor motion
            ego_truth (SubTrajectory): Actual ground truth motion the ego agent took
            frames (int): number of frames to predict into the future
            velocity_calc_method (VelocityCalc): methods used to calculate velocity if that is needed
                                                 for the algorithm
        Returns:
            Tuple[float, float]: Average displacement error, final displacement error
        """

        predicted = self.predict(ego_history, neighbor_history, frames, velocity_calc_method)

        difference = predicted - ego_truth
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

            
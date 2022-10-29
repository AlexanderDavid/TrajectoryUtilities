from .abstract import Predictor, SubTrajectory, VelocityCalc
from ..dataset import Position
from typing import Dict
from abc import abstractmethod

class LinearPredictor(Predictor):
    @abstractmethod
    def predict(self, ego_history: SubTrajectory, neighbor_history: Dict[int, SubTrajectory], frames: int,
                velocity_calc_method: VelocityCalc) -> SubTrajectory:
        ts = ego_history[0][1].time - ego_history[1][1].time

        velocity = Predictor.calc_velocity(velocity_calc_method)

        prediction = [
            Position(
                ego_history[0][-1].pos + i * velocity,
                None,
                ego_history[0][-1].time + i * ts
            ) for i in range(1, len(neighbor_history) + 1)
        ]

        return prediction, ego_history[1]
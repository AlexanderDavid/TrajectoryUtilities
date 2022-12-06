from .abstract import Predictor, VelocityCalc
from ..dataset import Position, Agent
from typing import List
import numpy as np

class NoisyLinearPredictor(Predictor):
    def __init__(self, frames: int, velocity_calc_method: VelocityCalc, n_preds: int=4, std_dev_degrees: float=25):
        super().__init__(frames, velocity_calc_method)
        self.std_dev = std_dev_degrees
        self.n_preds = n_preds
        
    def predict(
        self, ego_history: Agent, neighbors_history: List[Agent]
    ) -> List[List[Position]]:
        np.random.seed(1)
        positions = ego_history.positions
        ts = positions[1].time - positions[0].time

        predictions = []

        velocity = Predictor.calc_velocity(positions, self._velocity_calc_method)

        for noise in np.random.normal(0, self.std_dev, self.n_preds):
            print(noise)
            rad = np.radians(noise)
            r = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

            noisy_vel = r @ velocity

            predictions.append([
                Position(
                    positions[-1].pos + i * noisy_vel * ts,
                    noisy_vel,
                    positions[-1].time + i * ts,
                )
                for i in range(1, self._frames + 1)
            ])

        return predictions

class LinearPredictor(Predictor):
    def __init__(self, frames: int, velocity_calc_method: VelocityCalc):
        super().__init__(frames, velocity_calc_method)

    def predict(
        self, ego_history: Agent, neighbors_history: List[Agent]
    ) -> List[List[Position]]:
        positions = ego_history.positions
        ts = positions[1].time - positions[0].time

        velocity = Predictor.calc_velocity(positions, self._velocity_calc_method)

        predictions = [
            Position(
                positions[-1].pos + i * velocity * ts,
                velocity,
                positions[-1].time + i * ts,
            )
            for i in range(1, self._frames + 1)
        ]

        return [predictions]

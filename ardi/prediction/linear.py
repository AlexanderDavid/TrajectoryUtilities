from .abstract import Predictor, VelocityCalc
from ..dataset import Position, Agent
from typing import List


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

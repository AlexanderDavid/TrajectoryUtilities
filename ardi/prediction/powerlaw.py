from .abstract import Predictor, PrefVelocityCalc, VelocityCalc
from ..dataset import Position, Agent
from typing import List
from Powerlaw import PySimulationEngine
import numpy as np


class PowerlawPredictor(Predictor):
    def __init__(
        self,
        frames: int,
        velocity_calc_method: VelocityCalc,
        pref_velocity_calc_method: PrefVelocityCalc,
    ):
        super().__init__(frames, velocity_calc_method)

        self._pref_velocity_calc_method = pref_velocity_calc_method

    def predict(
        self, ego_history: Agent, neighbors_history: List[Agent]
    ) -> List[List[Position]]:
        engine = PySimulationEngine(500, 500, 1 / 60.0, 6000)
        last_velocity = Predictor.calc_velocity(
            ego_history.positions, self._velocity_calc_method
        )

        ego_idx = engine.addAgent(
            tuple(ego_history.positions[-1].pos),
            tuple(ego_history.goal),
            tuple(last_velocity),
            ego_history.radius,
            # TODO: Bad magic numbers
            1.3,
            20,
            0.5,
            10,
            1.5,
            0.54,
            2,
            3,
        )

        goal_vel = Predictor.calc_pref_velocity(
            ego_history.positions,
            ego_history,
            self._pref_velocity_calc_method,
            self._velocity_calc_method,
        )
        engine.setAgentPrefVelocity(ego_idx, tuple(goal_vel))

        neigh_map = {}
        for n in neighbors_history:
            last_velocity = Predictor.calc_velocity(
                n.positions, self._velocity_calc_method
            )

            neigh_map[n.idx] = (
                engine.addAgent(
                    tuple(n.positions[-1].pos),
                    tuple(n.goal),
                    tuple(last_velocity),
                    n.radius,
                    # TODO: Bad magic numbers
                    1.3,
                    20,
                    0.5,
                    10,
                    1.5,
                    0.54,
                    2,
                    3,
                ),
                n,
            )

            goal_vel = Predictor.calc_pref_velocity(
                n.positions,
                n,
                self._pref_velocity_calc_method,
                self._velocity_calc_method,
            )
            engine.setAgentPrefVelocity(neigh_map[n.idx][0], tuple(goal_vel))

        predictions = []
        for _ in range(self._frames):
            # TODO: Hardcode is badcode
            for _ in range(24):
                engine.updateSimulation()

            predictions.append(np.array(engine.getAgentPosition(ego_idx)))

            goal_vel = Predictor.calc_pref_velocity(
                ego_history.positions,
                ego_history,
                self._pref_velocity_calc_method,
                self._velocity_calc_method,
                current_pos=np.array(engine.getAgentPosition(ego_idx)),
            )
            engine.setAgentPrefVelocity(ego_idx, tuple(goal_vel))

            for n_idx in neigh_map:
                pl_idx, n = neigh_map[n_idx]
                goal_vel = Predictor.calc_pref_velocity(
                    n.positions,
                    n,
                    self._pref_velocity_calc_method,
                    self._velocity_calc_method,
                    current_pos=np.array(engine.getAgentPosition(pl_idx)),
                )
                engine.setAgentPrefVelocity(n_idx, tuple(goal_vel))

        return [predictions]

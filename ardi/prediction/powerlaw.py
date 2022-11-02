from .abstract import Predictor, PrefVelocityCalc, VelocityCalc
from ..dataset import Position, Agent
from typing import List
from Powerlaw import PySimulationEngine
import numpy as np


class PowerlawParams:
    def __init__(
        self,
        goal_radius: float = 0.25,
        neighbor_dist: float = 10,
        k: float = 1.5,
        ksi: float = 0.54,
        m: float = 2,
        t0: float = 3,
    ):
        self.goal_radius = goal_radius
        self.neighbor_dist = neighbor_dist
        self.k = k
        self.ksi = ksi
        self.m = m
        self.t0 = t0


class PowerlawPredictor(Predictor):
    def __init__(
        self,
        frames: int,
        velocity_calc_method: VelocityCalc,
        pref_velocity_calc_method: PrefVelocityCalc,
        powerlaw_fps: float = 1 / 60.0,
        powerlaw_skip: int = 24,
        powerlaw_params: PowerlawParams = PowerlawParams(),
    ):
        super().__init__(frames, velocity_calc_method)

        self._pref_velocity_calc_method = pref_velocity_calc_method
        self._powerlaw_fps = powerlaw_fps
        self._powerlaw_skip = powerlaw_skip
        self._powerlaw_params = powerlaw_params

    def predict(
        self, ego_history: Agent, neighbors_history: List[Agent]
    ) -> List[List[Position]]:
        engine = PySimulationEngine(500, 500, self._powerlaw_fps, 1000)
        last_velocity = Predictor.calc_velocity(
            ego_history.positions, self._velocity_calc_method
        )

        ego_idx = engine.addAgent(
            tuple(ego_history.positions[-1].pos),
            tuple(ego_history.goal * 1000),
            tuple(last_velocity),
            ego_history.radius,
            # TODO: Bad magic numbers
            1.3,
            self._powerlaw_params.goal_radius,
            self._powerlaw_params.neighbor_dist,
            self._powerlaw_params.k,
            self._powerlaw_params.ksi,
            self._powerlaw_params.m,
            self._powerlaw_params.t0,
        )

        goal_vel = Predictor.calc_pref_velocity(
            ego_history.positions,
            ego_history,
            self._pref_velocity_calc_method,
            self._velocity_calc_method,
        )
        engine.setAgentPrefVelocity(ego_idx, tuple(goal_vel))
        print(ego_history.positions[-1].pos)
        print(last_velocity)
        print(goal_vel)

        neigh_map = {}
        for n in neighbors_history:
            last_velocity = Predictor.calc_velocity(
                n.positions, self._velocity_calc_method
            )

            neigh_map[n.idx] = (
                engine.addAgent(
                    tuple(n.positions[-1].pos),
                    tuple(n.goal * 1000),
                    tuple(last_velocity),
                    n.radius,
                    # TODO: Bad magic numbers
                    0.5 if n.idx == -1 else 1.3,
                    self._powerlaw_params.goal_radius,
                    self._powerlaw_params.neighbor_dist,
                    self._powerlaw_params.k,
                    self._powerlaw_params.ksi,
                    self._powerlaw_params.m,
                    self._powerlaw_params.t0,
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

        ts = ego_history.positions[1].time - ego_history.positions[0].time

        predictions = []
        for i in range(self._frames):
            for _ in range(self._powerlaw_skip):
                engine.updateSimulation()

            predictions.append(
                Position(
                    np.array(engine.getAgentPosition(ego_idx)),
                    np.array(engine.getAgentVelocity(ego_idx)),
                    ego_history.positions[-1].time + (i + 1) * ts,
                )
            )

            goal_vel = Predictor.calc_pref_velocity(
                ego_history.positions,
                ego_history,
                self._pref_velocity_calc_method,
                self._velocity_calc_method,
                current_pos=Position(
                    np.array(engine.getAgentPosition(ego_idx)), None, None
                ),
            )
            engine.setAgentPrefVelocity(ego_idx, tuple(goal_vel))

            for n_idx in neigh_map:
                pl_idx, n = neigh_map[n_idx]
                goal_vel = Predictor.calc_pref_velocity(
                    n.positions,
                    n,
                    self._pref_velocity_calc_method,
                    self._velocity_calc_method,
                    current_pos=Position(
                        np.array(engine.getAgentPosition(pl_idx)), None, None
                    ),
                )
                engine.setAgentPrefVelocity(pl_idx, tuple(goal_vel))

        return [predictions]

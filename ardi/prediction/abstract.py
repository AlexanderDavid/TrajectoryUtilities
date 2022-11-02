from abc import ABC, abstractmethod
from enum import Enum
from ..dataset import Position, Agent
from typing import List, Optional, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects as mpe


class VelocityCalc(Enum):
    LAST_GROUND_TRUTH = 1
    AVERAGE_GROUND_TRUTH = 2
    LAST_DISPLACEMENT = 3
    AVERAGE_DISPLACEMENT = 4


class PrefVelocityCalc(Enum):
    FIX_MAG_ORACLE_DIR = 1
    FIX_MAG_INFER_DIR = 2
    INFER_MAG_ORACLE_DIR = 3
    INFER_MAG_INFER_DIR = 4


class Predictor(ABC):
    def __init__(self, frames: int, velocity_calc_method: VelocityCalc):
        self._frames = frames
        self._velocity_calc_method = velocity_calc_method

    @abstractmethod
    def predict(
        self, ego_history: Agent, neighbors_history: List[Agent]
    ) -> List[List[Position]]:
        """Predict ego trajectory into the future

        Args:
            ego_history (Agent): ego agent with the history in the positions field
            neighbors_history (List[Agent]): neighbors with the history in the positions field

        Returns:
            List[Position]: prediction for the ego agent
        """

    @staticmethod
    def ade_fde(
        ego_truth: List[Position], ego_preds: List[List[Position]]
    ) -> Tuple[float, float]:
        """Calculate the Average Displacement Error or Final Displacement Erorr

        Args:
            ego_truth (List[Position]): ground truth
            ego_pred (List[List[Position]]): list of predicted vels

        Returns:
            Tuple[float, float]: ADE, FDE
        """

        ade = np.inf
        fde = np.inf

        truth = np.array([x.pos for x in ego_truth])
        for pred in ego_preds:
            p = np.array([x.pos for x in pred])

            dists = [np.linalg.norm(x) for x in truth - p]

            ade = min(ade, np.mean(dists))
            fde = min(fde, dists[-1])

        return ade, fde

    @staticmethod
    def plot(
        ego_obs: List[Position],
        neighbors_obs: List[List[Position]],
        ego_truth: List[Position],
        neighbors_truth: List[List[Position]],
        predictions: List[List[Position]],
    ):
        plt.gca().set_aspect("equal")

        plt.plot(
            [x.pos[0] for x in ego_obs],
            [x.pos[1] for x in ego_obs],
            color="b",
            alpha=0.65,
            marker="o",
            markersize=3.2,
            linewidth=2.25,
            path_effects=[
                mpe.Stroke(linewidth=3, foreground="k", alpha=0.65),
                mpe.Normal(),
            ],
        )

        for pred in predictions:
            plt.plot(
                [p.pos[0] for p in pred],
                [p.pos[1] for p in pred],
                color="y",
                alpha=0.65,
                marker="o",
                markersize=3.2,
                linewidth=2.25,
                path_effects=[
                    mpe.Stroke(linewidth=3, foreground="k", alpha=0.65),
                    mpe.Normal(),
                ],
            )

        plt.plot(
            [p.pos[0] for p in ego_truth],
            [p.pos[1] for p in ego_truth],
            color="r",
            alpha=0.65,
            marker="o",
            markersize=3.2,
            linewidth=2.25,
            path_effects=[
                mpe.Stroke(linewidth=3, foreground="k", alpha=0.65),
                mpe.Normal(),
            ],
        )

        for neighbor_obs, neighbor_truth in zip(neighbors_obs, neighbors_truth):
            plt.plot(
                [x.pos[0] for x in neighbor_obs],
                [x.pos[1] for x in neighbor_obs],
                color="g",
                alpha=0.65,
                marker="o",
                markersize=3.2,
                linewidth=2.25,
                path_effects=[
                    mpe.Stroke(linewidth=3, foreground="k", alpha=0.65),
                    mpe.Normal(),
                ],
            )

            plt.plot(
                [x.pos[0] for x in neighbor_truth],
                [x.pos[1] for x in neighbor_truth],
                color="g",
                alpha=0.5,
                marker="o",
                markersize=3.2,
                linewidth=2.25,
                linestyle="dashed",
                path_effects=[
                    mpe.Stroke(linewidth=3, foreground="k", alpha=0.65),
                    mpe.Normal(),
                ],
            )

    @staticmethod
    def calc_velocity(
        poss: List[Position],
        method: VelocityCalc = VelocityCalc.LAST_GROUND_TRUTH,
        sampled_fps: int = 2.5,
    ) -> np.ndarray:
        """Calculate the velocity for an agent over a path

        Args:
            poss (List[Position]): History of the agent
            method (VelocityCalc, optional): Method of calculation to use. Either use the
                                             LAST or AVERAGE values for either the GROUND_TRUTH (velocity
                                             from each position) or DISPLACEMENT (infer velocity from difference
                                             between positions) . Defaults to VelocityCalc.LAST_GROUND_TRUTH.

        Raises:
            ValueError: If method is not a VelocityCalc

        Returns:
            np.ndarray: Calculated velocity
        """
        if method == VelocityCalc.LAST_GROUND_TRUTH:
            return poss[-1].vel

        if method == VelocityCalc.AVERAGE_GROUND_TRUTH:
            return np.mean([p.vel for p in poss], axis=0)

        if method == VelocityCalc.LAST_DISPLACEMENT:
            return (poss[-1].pos - poss[-2].pos) * sampled_fps

        if method == VelocityCalc.AVERAGE_DISPLACEMENT:
            positions = np.array([p.pos for p in poss])
            displacements = (positions[1:] - positions[:-1]) * sampled_fps
            return np.mean(displacements, axis=0)

        raise ValueError("method must be a valid VelocityCalculation")

    @staticmethod
    def calc_pref_velocity(
        poss: List[Position],
        ego: Agent,
        pref_method: PrefVelocityCalc = PrefVelocityCalc.FIX_MAG_ORACLE_DIR,
        vel_method: VelocityCalc = VelocityCalc.LAST_DISPLACEMENT,
        fixed_mag: float = 1.3,
        current_pos: Optional[Position] = None,
    ) -> np.ndarray:
        """Calculate the preferred velocity of an agent.

        Args:
            poss (List[Position]): History of the agent, used if the direction or magnitude ore inferred
            ego (Agent): Agent to calculate the preferred velocity of
            pref_method (PrefVelocityCalc, optional): Method of calculating the preferred velocity. There is a choice
                                                      between either fixing or inferring both the magnitude and direction.
                                                      A fixed magnitude will set the magnitude to some default, where inferring
                                                      it will use the velocity calculation method to get this value. Similar for
                                                      the direction. Defaults to PrefVelocityCalc.FIX_MAG_ORACLE_DIR.
            vel_method (VelocityCalc, optional): Method for calculating the velocity when inferring the pref speed from historical data.
                                                 Defaults to VelocityCalc.LAST_DISPLACEMENT.
            fixed_mag (float, optional): Fixed magnitude to use. Defaults to 1.3.
            current_pos (Optional[Position], optional): Current position if the oracle is used for the direction. Defaults to None.

        Returns:
            np.ndarray: Preferred speed for ego agent
        """
        if current_pos is None:
            current_pos = poss[-1]

        if pref_method == PrefVelocityCalc.FIX_MAG_ORACLE_DIR:
            pref_vel = ego.goal - current_pos.pos
            pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= fixed_mag

            return pref_vel

        if pref_method == PrefVelocityCalc.FIX_MAG_INFER_DIR:
            pref_vel = Predictor.calc_velocity(poss, vel_method)

            pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= fixed_mag

            return pref_vel

        if pref_method == PrefVelocityCalc.INFER_MAG_ORACLE_DIR:
            pref_vel = ego.goal - current_pos.pos
            pref_vel /= np.linalg.norm(pref_vel)
            pref_vel *= np.linalg.norm(Predictor.calc_velocity(poss, vel_method))

            return pref_vel

        if pref_method == PrefVelocityCalc.INFER_MAG_INFER_DIR:
            return Predictor.calc_velocity(poss, vel_method)

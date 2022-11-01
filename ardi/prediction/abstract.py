from abc import ABC, abstractmethod
from enum import Enum
from ..dataset import Position, Agent
from typing import List, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects as mpe


class VelocityCalc(Enum):
    LAST_GROUND_TRUTH = 1
    AVERAGE_GROUND_TRUTH = 2
    LAST_DISPLACEMENT = 3
    AVERAGE_DISPLACEMENT = 4


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
        poss: List[Position], method: VelocityCalc = VelocityCalc.LAST_GROUND_TRUTH
    ):
        if method == VelocityCalc.LAST_GROUND_TRUTH:
            return poss[-1].vel

        if method == VelocityCalc.AVERAGE_GROUND_TRUTH:
            return np.mean([p.vel for p in poss], axis=0)

        if method == VelocityCalc.LAST_DISPLACEMENT:
            return poss[-1].pos - poss[-2].pos

        if method == VelocityCalc.AVERAGE_DISPLACEMENT:
            positions = np.array([p.pos for p in poss])
            displacements = (positions[1:] - positions[:-1]) / (
                poss[1].time - poss[0].time
            )
            return np.mean(displacements, axis=0)

        raise ValueError("method must be a valid VelocityCalculation")

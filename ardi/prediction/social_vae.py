import numpy as np
from .abstract import Predictor, VelocityCalc
from ..dataset import Position, Agent
from .models import SocialVAE
import torch
from typing import List


class SocialVAEPredictor(Predictor):
    def __init__(
        self,
        ckpt_fn: str,
        ob_radius: float,
        n_predictions: int,
        frames: int,
        velocity_calc_method: VelocityCalc,
        device: str = "cpu",
    ):
        super().__init__(frames, velocity_calc_method)

        self._device = torch.device(device)

        self._model = SocialVAE(horizon=frames, ob_radius=ob_radius, hidden_dim=256)
        state_dict = torch.load(ckpt_fn, map_location=device)
        self._model.load_state_dict(state_dict["model"])
        self._model.to(self._device)
        self._model.eval()

        self._n_predictions = n_predictions

    @staticmethod
    def __get_agent_tensor(poss: List[Position]):
        vels = np.diff([x.pos for x in poss], axis=0)
        accs = np.diff(vels, axis=0)

        vels = np.concatenate(([vels[0]], vels), axis=0)
        accs = np.concatenate(([[0, 0], [0, 0]], accs), axis=0)
        return np.concatenate((np.array([x.pos for x in poss]), vels, accs), -1)

    @staticmethod
    def __get_neighbor_tensor(
        neighbor_poss: List[Position], agent_poss: List[Position]
    ):
        vels = np.subtract(
            [x.pos for x in neighbor_poss][1:], [x.pos for x in agent_poss][:-1]
        )
        accs = np.diff(vels, axis=0)

        vels = np.concatenate(([vels[0]], vels), axis=0)
        accs = np.concatenate(([[0, 0], [0, 0]], accs), axis=0)
        return np.concatenate(
            (np.array([x.pos for x in neighbor_poss]), vels, accs), -1
        )

    def predict(
        self, ego_history: Agent, neighbors_history: List[Agent]
    ) -> List[List[Position]]:
        SocialVAE.seed(1)
        neighbor_numpy = []

        for neigh in neighbors_history:
            neighbor_numpy.append(
                SocialVAEPredictor.__get_neighbor_tensor(
                    neigh.positions, ego_history.positions
                )
            )

        if len(neighbor_numpy):
            neighbor_numpy = np.hstack(neighbor_numpy).reshape(
                len(ego_history.positions), len(neighbors_history), 6
            )
            # neighbor_numpy = np.hstack(neighbor_numpy).reshape(4, 1,  6)
            neighbor_tensor = (
                torch.from_numpy(neighbor_numpy)
                .to(torch.float32)
                .unsqueeze(1)
                .to(self._device)
            )
        else:
            neighbor_numpy = np.zeros((len(neighbors_history), 0, 6))
            neighbor_tensor = (
                torch.from_numpy(neighbor_numpy)
                .to(torch.float32)
                .unsqueeze(1)
                .to(self._device)
            )

        ego_tensor = (
            torch.from_numpy(
                SocialVAEPredictor.__get_agent_tensor(ego_history.positions)
            )
            .to(torch.float32)
            .unsqueeze(1)
            .to(self._device)
        )

        preds = (
            self._model(ego_tensor, neighbor_tensor, n_predictions=2000)
            .squeeze(2)
            .detach()
            .cpu()
            .numpy()
        )
        preds = preds[SocialVAE.FPC(preds, n_samples=self._n_predictions)]

        poss = []
        for pred in preds:
            poss.append([Position(p, np.array([0, 0]), 0) for p in pred])

        return poss

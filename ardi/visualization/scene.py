from random import choice
from typing import Tuple, Optional, List
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import cv2 as cv


class Agent:
    SHIRT = 255, 0, 0,
    SKIN = 0, 255, 0
    HAIR = 0, 0, 255

    SKINS = [
        [85, 79, 72],
        [68, 59, 49],
        # [37, 29, 22]
    ]

    SHIRTS = [
        [230, 173, 236],
        [255, 111, 89],
        [37, 68, 65],
    ]

    HAIRS = [
        [5, 5, 5],
        [255, 175, 135],
        [120, 79, 75]
    ]

    IMG_PATH = Path(__file__).parent / "guy.png"

    def __init__(
        self,
        pos: Tuple[float, float],
        goal: Tuple[float, float],
        goal_oriented: bool = True,
        orientation: Optional[float] = None,
    ):
        self._pos = np.array(pos)
        self._goal = np.array(goal)
        if goal_oriented:
            v1 = self._pos / np.linalg.norm(self._pos)
            v2 = self._goal / np.linalg.norm(self._goal)

            self._orientation = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        else:
            self._orientation = orientation

        img = np.asarray(Image.open(Agent.IMG_PATH))
        self._img = self.__generate_random_guy(img)

    @staticmethod
    def __generate_random_guy(img: np.array) -> np.array:
        img_ = np.copy(img)

        img_ = Agent.__change_color(img_, Agent.HAIR, choice(Agent.HAIRS))
        img_ = Agent.__change_color(img_, Agent.SHIRT, choice(Agent.SHIRTS))
        img_ = Agent.__change_color(img_, Agent.SKIN, choice(Agent.SKINS))

        return img_

    @staticmethod
    def __change_color(
        img: np.array, find: Tuple[int, int, int], replace: Tuple[int, int, int]
    ) -> np.array:
        img_ = np.copy(img)

        r, g, b = img_[:, :, 0], img[:, :, 1], img[:, :, 2]
        mask = (find[0] == r) & (find[1] == g) & (find[2] == b)
        img_[:, :, :3][mask] = [*replace]

        return img_

class Scene:

    def __init__(self, agents: List[Agent]):
        self._agents = agents

    def __init__(self, poss: List[Tuple[float, float]], goals: List[Tuple[float, float]]):
        self._agents = [
            Agent(pos, goal, True) for pos, goal in zip(poss, goals)
        ]

    def show(self):
        

    def save(self):
        pass

from random import choice
from typing import Tuple, Optional, List
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


class Agent:
    SHIRT = (
        255,
        0,
        0,
    )
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

    HAIRS = [[5, 5, 5], [255, 175, 135], [120, 79, 75]]

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

    @property
    def pos(self):
        return self._pos

    @property
    def goal(self):
        return self._goal

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

    def __init__(
        self, poss: List[Tuple[float, float]], goals: List[Tuple[float, float]]
    ):
        self._agents = [Agent(pos, goal, True) for pos, goal in zip(poss, goals)]

    def __get_extents(self) -> Tuple[float, float, float, float]:
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for a in self._agents:
            max_x = max([a.pos[0], a.goal[0], max_x])
            max_y = max([a.pos[1], a.goal[1], max_y])

            min_x = min([a.pos[0], a.goal[0], min_x])
            min_y = min([a.pos[1], a.goal[1], min_y])

        return min_x, min_y, max_x, max_y

    def show(self):
        agent_patches = [
            Circle(a.pos, 0.2, facecolor="blue", edgecolor="black")
            for a in self._agents
        ]

        goal_patches = [
            Circle(a.goal, 0.2, facecolor="red", edgecolor="black")
            for a in self._agents
        ]

        min_x, min_y, max_x, max_y = self.__get_extents()

        x_range = abs(max_x - min_x)
        y_range = abs(max_y - min_y)

        x_border = x_range * 0.1
        y_border = y_range * 0.1

        if x_border == 0:
            x_border = 2
        if y_border == 0:
            y_border = 2

        fig, ax = plt.subplots()

        for gp, ap in zip(goal_patches, agent_patches):
            ax.add_patch(gp)
            ax.add_patch(ap)

        ax.set_xlim((min_x - x_border, max_x + x_border))
        ax.set_ylim((min_y - y_border, max_y + y_border))

        plt.show()

    def save(self):
        pass

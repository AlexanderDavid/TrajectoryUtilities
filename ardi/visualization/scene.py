from random import choice
from typing import Tuple, Optional, List
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arrow, FancyBboxPatch


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

    SHIRT_IDX = 0
    SHIRTS = [
        [230, 173, 236],
        [255, 111, 89],
        [37, 68, 65],
        [0, 0, 0]
    ]

    HAIRS = [[5, 5, 5], [255, 175, 135], [120, 79, 75]]

    IMG_PATH = Path(__file__).parent / "guy.png"

    def __init__(
        self,
        pos: Tuple[float, float],
        goal: Tuple[float, float],
        radius: float = 0.3,
        goal_oriented: bool = True,
        orientation: Optional[float] = None,
    ):
        self._pos = np.array(pos, dtype=np.float32)
        self._goal = np.array(goal, dtype=np.float32)
        self._radius = radius
        if goal_oriented:
            self._vel = self._goal - self.pos
            self._vel += np.finfo(float).eps
            self._vel /= np.linalg.norm(self._vel)
            self._vel *= self._radius * 2
        else:
            # self._orientation = orientation
            raise NotImplementedError()

        self._main_color = None
        img = np.asarray(Image.open(Agent.IMG_PATH))
        self._img = self.__generate_random_guy(img)

    @property
    def pos(self):
        return self._pos

    @property
    def goal(self):
        return self._goal

    @property
    def radius(self):
        return self._radius

    @property
    def vel(self):
        return self._vel

    @property
    def color(self):
        return self._main_color

    def __generate_random_guy(self, img: np.array) -> np.array:
        img_ = np.copy(img)

        self._main_color = Agent.SHIRTS[Agent.SHIRT_IDX]
        Agent.SHIRT_IDX += 1

        img_ = Agent.__change_color(img_, Agent.HAIR, choice(Agent.HAIRS))
        img_ = Agent.__change_color(img_, Agent.SHIRT, self._main_color)
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
            max_x = max([a.pos[0] + a.radius, a.goal[0] + a.radius, max_x])
            max_y = max([a.pos[1] + a.radius, a.goal[1] + a.radius, max_y])

            min_x = min([a.pos[0] - a.radius, a.goal[0] - a.radius, min_x])
            min_y = min([a.pos[1] - a.radius, a.goal[1] - a.radius, min_y])

        return min_x, min_y, max_x, max_y

    def show(self):
        agent_patches = [
            Circle(a.pos, a.radius, facecolor=np.array(a.color) / 255, edgecolor="black")
            for a in self._agents
        ]

        arrow_patches = [
            Arrow(a.pos[0], a.pos[1], a.vel[0], a.vel[1], facecolor=np.array(a.color) / 255, edgecolor="black", lw=2) for a in self._agents
        ]

        inner_arrow_patches = [
            Arrow(a.pos[0], a.pos[1], a.vel[0], a.vel[1], facecolor=np.array(a.color) / 255) for a in self._agents
        ]

        goal_patches = [
            FancyBboxPatch(a.goal - a.radius / 2, a.radius, a.radius, facecolor=np.array(a.color) / 255, edgecolor="black")
            for a in self._agents
        ]

        min_x, min_y, max_x, max_y = self.__get_extents()

        x_range = abs(max_x - min_x)
        y_range = abs(max_y - min_y)

        x_border = x_range * 0.1
        y_border = y_range * 0.1

        _, ax = plt.subplots()
        ax.axis("off")

        for gp, ap, arr_p, ip in zip(goal_patches, agent_patches, arrow_patches, inner_arrow_patches):
            ax.add_patch(gp)
            ax.add_patch(arr_p)
            ax.add_patch(ap)
            ax.add_patch(ip)

        ax.set_xlim((min_x - x_border, max_x + x_border))
        ax.set_ylim((min_y - y_border, max_y + y_border))
        ax.set_aspect("equal")

        plt.show()

    def save(self):
        pass

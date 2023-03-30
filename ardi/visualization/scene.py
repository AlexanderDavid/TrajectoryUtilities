from copy import deepcopy
from random import choice
from typing import Tuple, Optional, List
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Circle, Arrow, FancyBboxPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class Agent:
    SHIRT = (
        255,
        0,
        0,
    )
    SKIN = 255, 0, 0
    HAIR = 0, 0, 255

    SKINS = [
        [85, 79, 72],
        [68, 59, 49],
        # [37, 29, 22]
    ]

    SHIRT_IDX = 0
    SHIRTS = [[218, 196, 247], [244, 152, 156], [235, 210, 180]]

    HAIRS = [[5, 5, 5], [255, 175, 135], [120, 79, 75]]

    IMG_PATH = Path(__file__).parent / "guy.png"
    ROBOT_PATH = Path(__file__).parent / "robot.png"

    def __init__(
        self,
        pos: Tuple[float, float],
        goal: Tuple[float, float],
        radius: float = 0.3,
        robot: bool = False,
    ):
        self._pos = np.array(pos, dtype=np.float32)
        self._goal = np.array(goal, dtype=np.float32)
        self._radius = radius

        self._vel = self._goal - self.pos
        self._vel += np.finfo(float).eps
        self._vel /= np.linalg.norm(self._vel)
        self._vel *= self._radius * 2
        self._orientation = np.arctan2(self._vel[1], self.vel[0])

        pil_img = Image.open(Agent.ROBOT_PATH if robot else Agent.IMG_PATH)
        pil_img = pil_img.rotate(
            np.degrees(self._orientation) - 90, Image.NEAREST, expand=1
        )
        img = np.asarray(pil_img)

        self._main_color = (0.1, 0.1, 0.1)
        self._img = img if robot else self.__generate_random_guy(img)
        self._offset_img = OffsetImage(self._img, zoom=0.125 if robot else 0.075)

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

    def get_agent_patches(self) -> List[Patch]:
        return [
            # Outline of arrow
            Arrow(
                self.pos[0],
                self.pos[1],
                self.vel[0],
                self.vel[1],
                facecolor=np.array(self.color) / 255,
                edgecolor="black",
                lw=2,
            ),
            # Main body
            Circle(
                self.pos,
                self.radius,
                facecolor=np.array(self.color) / 255,
                edgecolor="black",
            ),
            # Inside of arrow to fill circles border
            Arrow(
                self.pos[0],
                self.pos[1],
                self.vel[0],
                self.vel[1],
                facecolor=np.array(self.color) / 255,
            ),
        ]

    def get_goal_patch(self) -> Patch:
        return FancyBboxPatch(
            self.goal - self.radius / 2,
            self.radius,
            self.radius,
            facecolor=np.array(self.color) / 255,
            edgecolor="black",
        )

    def get_agent_image(self) -> AnnotationBbox:
        return AnnotationBbox(self._offset_img, self.pos, pad=0, frameon=False)

    def __generate_random_guy(self, img: np.array) -> np.array:
        img_ = np.copy(img)

        self._main_color = Agent.SHIRTS[Agent.SHIRT_IDX % len(Agent.SHIRTS)]
        Agent.SHIRT_IDX += 1

        img_ = Agent.__change_color(img_, Agent.SHIRT, self._main_color)

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
    def __init__(
        self,
        poss: List[Tuple[float, float]],
        goals: List[Tuple[float, float]],
        robot_idx: Optional[int],
        border_pct: float,
    ):
        self._agents = [
            Agent(pos, goal, True, robot_idx is not None and i == robot_idx)
            for i, (pos, goal) in enumerate(zip(poss, goals))
        ]
        self._border_pct = border_pct

    def __get_extents(self) -> Tuple[float, float, float, float]:
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for a in self._agents:
            max_x = max([a.pos[0] + a.radius, a.goal[0] + a.radius, max_x])
            max_y = max([a.pos[1] + a.radius, a.goal[1] + a.radius, max_y])

            min_x = min([a.pos[0] - a.radius, a.goal[0] - a.radius, min_x])
            min_y = min([a.pos[1] - a.radius, a.goal[1] - a.radius, min_y])

        return min_x, min_y, max_x, max_y

    def __generate_figure(self):
        # Find out where to focus
        min_x, min_y, max_x, max_y = self.__get_extents()
        x_range = abs(max_x - min_x)
        y_range = abs(max_y - min_y)
        x_border = x_range * self._border_pct
        y_border = y_range * self._border_pct


        # Generate the background checkerboard
        _, ax = plt.subplots()
        cell = 2.5
        for i, x in enumerate(np.arange(min_x, max_x, cell)):
            for j, y in enumerate(np.arange(min_y, max_y, cell)):
                color = (0.908, 0.908, 0.908)
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                    color = (0.97, 0.97, 0.97)

                ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color))

        # Plot the trajectory lines
        # for agent in self._agents:
        #     ax.plot([agent.pos[0], agent.goal[0]], [agent.pos[1], agent.goal[1]], color=np.array(agent.color) / 255, alpha=0.5)

        # Add the goal rounded rectangles
        for p in [a.get_goal_patch() for a in self._agents]:
            ax.add_patch(p)

        # Add the icons for the agents
        for artist in [a.get_agent_image() for a in self._agents]:
            ax.add_artist(artist)

        # Make the plots look plain and zoomed in
        ax.axis("off")
        ax.set_xlim((min_x - x_border, max_x + x_border))
        ax.set_ylim((min_y - y_border, max_y + y_border))
        ax.set_aspect("equal")

        return ax

    def show(self):
        self.__generate_figure()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

    def save(self, filename: str, format: str, dpi: int = 1200):
        self.__generate_figure()
        # plt.subplots_adjust(left=0, right=0.0001, top=0.0001, bottom=0)
        plt.tight_layout()
        plt.savefig(filename, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)

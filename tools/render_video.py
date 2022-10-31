#! /usr/bin/env python

import argparse
from pathlib import Path
from typing import Optional

from matplotlib import animation, collections
from matplotlib import pyplot as plt

from ardi.dataset import ZuckerDataset


class TrajectoryRenderer:
    def __init__(self, input_filename: Path, save_filename: Optional[Path]):
        print(f"Reading in {input_filename.resolve()}...")

        self._filename = input_filename
        self._data = ZuckerDataset(str(input_filename))
        self._save = save_filename is not None
        self._save_fn = save_filename

        # Calculate the extents of the trajectory and make it square
        self._min_x, self._max_x, self._min_y, self._max_y = self._data.extents

        self._min_x -= 1
        self._min_y -= 1
        self._max_x += 1
        self._max_y += 1

        y_range = abs(self._max_y - self._min_y)
        x_range = abs(self._max_x - self._min_x)

        if y_range > x_range:
            diff = y_range - x_range
            self._min_x -= diff / 2
            self._max_x += diff / 2
        else:
            diff = x_range - y_range
            self._min_y -= diff / 2
            self._max_y += diff / 2

    def render(self):
        fig = plt.figure()
        self._ax = plt.axes()

        self._ax.set_xlim([self._min_x, self._max_x])
        self._ax.set_ylim([self._min_y, self._max_y])
        self._ax.set_aspect("equal")

        ani = animation.FuncAnimation(
            fig, self._animate, interval=self._data.timestep * 1000, frames=len(self._data.times)
        )
        if self._save:
            writer = animation.FFMpegWriter(fps=1 / self._data.timestep)
            ani.save(str(self._save_fn), writer=writer)
        else:
            plt.show()

    def _animate(self, time_idx):
        self._ax.clear()
        self._ax.set_xlim([self._min_x, self._max_x])
        self._ax.set_ylim([self._min_y, self._max_y])

        self._ax.set_title(
            f"{self._filename.resolve()}\n"
            + f"Time = {str(round(self._data.times[time_idx], 3)).ljust(5, '0')}"
        )

        circles = []
        poss = self._data.get_positions(self._data.times[time_idx])
        for idx in poss:
            pos, agent = poss[idx]

            circles.append(plt.Circle(pos.pos, radius=agent.radius, color=agent.color))

        self._ax.add_collection(collections.PatchCollection(circles))


def main():
    parser = argparse.ArgumentParser(description="Render a trajectory file to an mp4")

    parser.add_argument("input_file", type=Path, help="Path to the trajectory file")

    parser.add_argument(
        "--save", type=Path, default=None, help="Path to save the animation to"
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: no such trajectory file ({str(args.input_file)}) exists")
        exit(1)

    TrajectoryRenderer(args.input_file, args.save)


if __name__ == "__main__":
    main()

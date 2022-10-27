#! /usr/bin/env python

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import animation, collections
from matplotlib import pyplot as plt

import seaborn as sns

sns.set_theme()


class TrajectoryRenderer:
    def __init__(self, input_filename: Path, fps: float, save_filename: Optional[Path]):
        print(f"Reading in {input_filename.resolve()}...")

        self._filename = input_filename
        self._data = pd.read_csv(
            str(input_filename), names=["id", "ts", "x", "y", "gx", "gy"], skiprows=[0]
        ).apply(pd.to_numeric)
        self._times = pd.unique(self._data.ts)
        self._idxs = pd.unique(self._data.id)

        # Calculate the extents of the trajectory and make it square
        self._min_x = self._data.x.min() - 1
        self._max_x = self._data.x.max() + 1
        self._min_y = self._data.y.min() - 1
        self._max_y = self._data.y.max() + 1

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

        fig = plt.figure()
        self._ax = plt.axes()

        self._ax.set_xlim([self._min_x, self._max_x])
        self._ax.set_ylim([self._min_y, self._max_y])
        self._ax.set_aspect("equal")

        ani = animation.FuncAnimation(
            fig, self._animate, interval=1 / fps * 1000, frames=len(self._times)
        )
        if save_filename:
            writer = animation.FFMpegWriter(fps=fps)
            ani.save(str(save_filename), writer=writer)
        else:
            plt.show()

    def _animate(self, time_idx):
        self._ax.clear()
        self._ax.set_xlim([self._min_x, self._max_x])
        self._ax.set_ylim([self._min_y, self._max_y])

        self._ax.set_title(
            f"{self._filename.resolve()}\nTime = {str(round(self._times[time_idx], 3)).ljust(5, '0')}"
        )

        circles = []
        for idx in self._idxs:
            pos = self._data[
                (self._data.id == idx) & (self._data.ts == self._times[time_idx])
            ][["x", "y"]].to_numpy()

            if len(pos) == 0:
                continue

            circles.append(plt.Circle(pos[0], radius=0.17 if idx == -1 else 0.1))

        self._ax.add_collection(collections.PatchCollection(circles))


def main():
    parser = argparse.ArgumentParser(description="Render a trajectory file to an mp4")

    parser.add_argument("input_file", type=Path, help="Path to the trajectory file")

    parser.add_argument(
        "--fps", type=float, default=60, help="Frames per second to render at"
    )
    parser.add_argument(
        "--save", type=Path, default=None, help="Path to save the animation to"
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: no such trajectory file ({str(args.input_file)}) exists")
        exit(1)

    TrajectoryRenderer(args.input_file, args.fps, args.save)


if __name__ == "__main__":
    main()

#! /usr/bin/env python 
 
from pathlib import Path
import argparse 

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import collections

class TrajectoryRenderer:
    def __init__(self, input_filename: Path):
        print(f"Reading in {input_filename.resolve()}...")
        
        self._filename = input_filename
        self._data = pd.read_csv(str(input_filename), names=["id", "ts", "x", "y", "gx", "gy"], skiprows=[0]).apply(pd.to_numeric)
        self._times = pd.unique(self._data.ts)
        self._idxs = pd.unique(self._data.id)

        # Calculate the extents of the trajectory
        self._min_x = self._data.x.min()
        self._max_x = self._data.x.max()
        self._min_y = self._data.y.min()
        self._max_y = self._data.y.max()

        fig = plt.figure()
        self._ax = plt.axes()

        self._ax.set_xlim([self._min_x, self._max_x])
        self._ax.set_ylim([self._min_y, self._max_y])

        ani = animation.FuncAnimation(fig, self._animate, interval=3, frames=len(self._times))
        plt.show()

    def _animate(self, time_idx):
        self._ax.clear()
        self._ax.set_xlim([self._min_x, self._max_x])
        self._ax.set_ylim([self._min_y, self._max_y])

        self._ax.set_title(f"{self._filename.resolve()}\nTime = {str(round(self._times[time_idx], 3)).ljust(5, '0')}")

        circles = []
        for idx in self._idxs:
            pos = self._data[(self._data.id == idx) & (self._data.ts == self._times[time_idx])][["x", "y"]].to_numpy()

            if len(pos) == 0:
                continue           

            circles.append(
                plt.Circle(pos[0], radius=0.17 if idx == -1 else 0.1)
            )

        self._ax.add_collection(collections.PatchCollection(circles))


def main():
    parser = argparse.ArgumentParser(description="Render a trajectory file to an mp4")

    parser.add_argument("input_file", type=Path, help="Path to the trajectory file")

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: no such trajectory file ({str(args.input_file)}) exists")
        exit(1)

    TrajectoryRenderer(args.input_file)

if __name__ == "__main__":
    main()
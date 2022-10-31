from typing import Union, List
from ..dataset import Agent

import numpy as np


def speed(ego: Agent, average: bool = True) -> Union[float, List[float]]:
    speeds = [np.linalg.norm(x.vel) for x in ego.positions]

    if average:
        return np.mean(speeds)

    return speeds

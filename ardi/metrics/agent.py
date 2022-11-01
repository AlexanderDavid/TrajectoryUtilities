from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, List

if TYPE_CHECKING:
    from ..dataset import Agent

import numpy as np


def speed(ego: Agent, average: bool = True) -> Union[float, List[float]]:
    speeds = [np.linalg.norm(x.vel) for x in ego.positions]

    if average:
        return np.mean(speeds)

    return speeds

import unittest

import numpy as np

from ardi.prediction import Predictor, VelocityCalc
from ardi.dataset import Position

class TestAbstractPredictionMethods(unittest.TestCase):
    def test_velocity_calc_last_ground_truth(self):
        self.assertEqual(
            Predictor.calc_velocity(
                [Position(np.array([0, 0]), np.array([1, 0]), 0)],
                VelocityCalc.LAST_GROUND_TRUTH
            ).tolist(),
            [1, 0]
        )
    
    def test_velocity_calc_average_ground_truth(self):
        self.assertEqual(
            Predictor.calc_velocity(
                [
                    Position(np.array([0, 0]), np.array([1, 0]), 0),
                    Position(np.array([0, 0]), np.array([0, 1]), 1),
                ],
                VelocityCalc.AVERAGE_GROUND_TRUTH
            ).tolist(),
            [0.5, 0.5]
        )

    def test_velocity_calc_last_displacement(self):
        self.assertEqual(
            Predictor.calc_velocity(
                [
                    Position(np.array([1, 0]), np.array([5, 0]), 0),
                    Position(np.array([0, 0]), np.array([5, 0]), 1),
                ],
                VelocityCalc.LAST_DISPLACEMENT
            ).tolist(),
            [1, 0]
        )

    def test_velocity_calc_average_displacement(self):
        self.assertEqual(
            Predictor.calc_velocity(
                [
                    Position(np.array([6, 0]), np.array([5, 0]), 0),
                    Position(np.array([3, 0]), np.array([5, 0]), 1),
                    Position(np.array([2, 0]), np.array([5, 0]), 2),
                    Position(np.array([0, 0]), np.array([5, 0]), 3),
                ],
                VelocityCalc.AVERAGE_DISPLACEMENT
            ).tolist(),
            [-2, 0]
        )
        
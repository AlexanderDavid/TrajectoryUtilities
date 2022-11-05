import unittest

import numpy as np

from ardi.prediction import Predictor, VelocityCalc, LinearPredictor
from ardi.dataset import Position, Agent
from ardi.prediction.social_vae import SocialVAEPredictor


class TestLinearPredictor(unittest.TestCase):
    def test_linear_prediction(self):
        lp = LinearPredictor(3, VelocityCalc.LAST_GROUND_TRUTH)

        ego = Agent(0, "", 0, np.array([0, 0]), np.array([0, 0]))
        ego.positions = [
            Position(np.array([0, 0]), np.array([1, 0]), 0),
            Position(np.array([1, 0]), np.array([1, 0]), 1),
            Position(np.array([2, 0]), np.array([1, 0]), 2),
            Position(np.array([3, 0]), np.array([1, 0]), 3),
        ]

        self.assertListEqual(
            lp.predict(ego, [])[0],
            [
                Position(np.array([4, 0]), np.array([1, 0]), 4),
                Position(np.array([5, 0]), np.array([1, 0]), 5),
                Position(np.array([6, 0]), np.array([1, 0]), 6),
            ],
        )

    def test_linear_prediction_backwards(self):
        lp = LinearPredictor(3, VelocityCalc.LAST_GROUND_TRUTH)

        ego = Agent(0, "", 0, np.array([0, 0]), np.array([0, 0]))
        ego.positions = [
            Position(np.array([-0, 0]), np.array([-1, 0.5]), 0),
            Position(np.array([-1, 0.5]), np.array([-1, 0.5]), 1),
            Position(np.array([-2, 1]), np.array([-1, 0.5]), 2),
            Position(np.array([-3, 1.5]), np.array([-1, 0.5]), 3),
        ]

        self.assertListEqual(
            lp.predict(ego, [])[0],
            [
                Position(np.array([-4, 2]), np.array([-1, 0.5]), 4),
                Position(np.array([-5, 2.5]), np.array([-1, 0.5]), 5),
                Position(np.array([-6, 3]), np.array([-1, 0.5]), 6),
            ],
        )


class TestSocialVAEPredictor(unittest.TestCase):
    def test_social_vae_prediction(self):
        svp = SocialVAEPredictor(
            "../SocialVAE/models/zara01/ckpt-best",
            4,
            5,
            8,
            VelocityCalc.LAST_DISPLACEMENT,
        )

        ego = Agent(0, "", 0, np.array([0, 0]), np.array([0, 0]))
        ego.positions = [
            Position(np.array([0, 0]), np.array([1, 0]), 0),
            Position(np.array([1, 0]), np.array([1, 0]), 1),
            Position(np.array([2, 0]), np.array([1, 0]), 2),
            Position(np.array([3, 0]), np.array([1, 0]), 3),
        ]

        neigh = Agent(0, "", 0, np.array([0, 0]), np.array([0, 0]))
        neigh.positions = [
            Position(np.array([10, 0]), np.array([-1, 0]), 0),
            Position(np.array([9, 0]), np.array([-1, 0]), 1),
            Position(np.array([8, 0]), np.array([-1, 0]), 2),
            Position(np.array([7, 0]), np.array([-1, 0]), 3),
        ]

        ps = svp.predict(ego, [neigh])[0]
        ts = [
            Position(np.array([3.9736335, 0.01945923]), np.array([0, 0]), None),
            Position(np.array([4.916696, 0.02303]), np.array([0, 0]), None),
            Position(np.array([5.8411818, 0.02792673]), np.array([0, 0]), None),
            Position(np.array([6.7932243, 0.05065208]), np.array([0, 0]), None),
            Position(np.array([7.7247143, 0.07876479]), np.array([0, 0]), None),
            Position(np.array([8.59561, 0.11718043]), np.array([0, 0]), None),
            Position(np.array([9.347851, 0.15412699]), np.array([0, 0]), None),
            Position(np.array([9.985252, 0.2505523]), np.array([0, 0]), None),
        ]

        for p, t in zip(ps, ts):
            self.assertTrue(np.allclose(p.pos, t.pos))


class TestAbstractPredictionMethods(unittest.TestCase):
    def test_velocity_calc_last_ground_truth(self):
        self.assertEqual(
            Predictor.calc_velocity(
                [Position(np.array([0, 0]), np.array([1, 0]), 0)],
                VelocityCalc.LAST_GROUND_TRUTH,
            ).tolist(),
            [1, 0],
        )

    def test_velocity_calc_average_ground_truth(self):
        self.assertEqual(
            Predictor.calc_velocity(
                [
                    Position(np.array([0, 0]), np.array([1, 0]), 0),
                    Position(np.array([0, 0]), np.array([0, 1]), 1),
                ],
                VelocityCalc.AVERAGE_GROUND_TRUTH,
            ).tolist(),
            [0.5, 0.5],
        )

    def test_velocity_calc_last_displacement(self):
        self.assertEqual(
            Predictor.calc_velocity(
                [
                    Position(np.array([1, 0]), np.array([5, 0]), 0),
                    Position(np.array([0, 0]), np.array([5, 0]), 1),
                ],
                VelocityCalc.LAST_DISPLACEMENT,
                1
            ).tolist(),
            [-1, 0],
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
                VelocityCalc.AVERAGE_DISPLACEMENT,
                1
            ).tolist(),
            [-2, 0],
        )

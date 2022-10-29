from ardi import metrics
from ardi.dataset import ZuckerDataset

import numpy as np

import unittest

class TestZuckerDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data = ZuckerDataset(
            "./tests/sample_data/zucker.csv"
        )
        return super().setUp()

    def test_timestep(self):
        self.assertAlmostEqual(self.data.timestep, 1/60., 5)

    def test_length(self):
        self.assertEqual(len(self.data.times), 560)

    def test_num_agents(self):
        self.assertEqual(len(self.data.agents), 3)

    def test_specific_positions(self):
        poss = self.data.get_positions(self.data.times[0])

        self.assertEqual(poss[-1][0].pos.tolist(), [-1.832912, 0.528822])
        self.assertEqual(poss[3][0].pos.tolist(), [6.40716, 1.426972])
        self.assertEqual(poss[4][0].pos.tolist(), [6.378485, 2.262925])


if __name__ == "__main__":
    unittest.main()
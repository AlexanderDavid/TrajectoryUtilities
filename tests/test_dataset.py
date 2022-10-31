from ardi.dataset import ZuckerDataset, SocialVAEDataset

import numpy as np

import unittest


class TestSocialVAEDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data = SocialVAEDataset("./tests/sample_data/socialvae.txt")
        return super().setUp()

    def test_length(self):
        self.assertEqual(len(self.data.times), 21)

    def test_timestep(self):
        self.assertAlmostEqual(self.data.timestep, 1, 5)

    def test_num_agents(self):
        self.assertEqual(len(self.data.agents.keys()), 2)

    def test_specific_positions(self):
        poss = self.data.get_positions(self.data.times[0])

        self.assertEqual(poss[6][0].pos.tolist(), [5.479989, 0.826256])
        self.assertEqual(poss[-1][0].pos.tolist(), [2.950434, 0.516553])


class TestZuckerDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data = ZuckerDataset("./tests/sample_data/zucker.csv")
        return super().setUp()

    def test_timestep(self):
        self.assertAlmostEqual(self.data.timestep, 1 / 60.0, 5)

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

from ardi import metrics
from ardi.dataset import Position

import numpy as np
import unittest

class TestMetrics(unittest.TestCase):
    def test_ttca_facing_away(self):
        """Ensure that the ttca is negative when agents are facing away from eachother
        """
        self.assertEqual(
            metrics.ttca(
                Position(np.array([0, 0]), np.array([1, 0]), 0),
                Position(np.array([-1, 0]), np.array([-1, 0]), 0),
            ),
            -0.5
        )

    def test_ttca_direct_collision(self):
        """Ensure ttca is at the moment of a direct collision
        """
        self.assertEqual(
            metrics.ttca(
                Position(np.array([0, 0]), np.array([1, 0]), 0),
                Position(np.array([1, 0]), np.array([-1, 0]), 0),
            ),
            0.5
        )

    def test_ttca_never_touch_getting_closer(self):
        self.assertEqual(
            metrics.ttca(
                Position(np.array([0, 1]), np.array([1, 0]), 0),
                Position(np.array([1, 0]), np.array([-1, 0]), 0),
            ),
            0.5
        )

    def test_ttca_never_touch_getting_further_away(self):
        self.assertEqual(
            metrics.ttca(
                Position(np.array([0, 1]), np.array([-1, 0]), 0),
                Position(np.array([1, 0]), np.array([1, 0]), 0),
            ),
            -0.5
        )
        
    def test_mpd_facing_away(self):
        """Ensure that the mpd is the current distance when facing away from eachother
        """
        self.assertEqual(
            metrics.mpd(
                Position(np.array([0, 0]), np.array([1, 0]), 0),
                Position(np.array([-1, 0]), np.array([-1, 0]), 0),
            ),
            1
        )

    def test_mpd_direct_collision(self):
        """Ensure mpd is at the moment of a direct collision if one exists
        """
        self.assertEqual(
            metrics.mpd(
                Position(np.array([0, 0]), np.array([1, 0]), 0),
                Position(np.array([1, 0]), np.array([-1, 0]), 0),
            ),
            0
        )

    def test_mpd_never_touch_getting_closer(self):
        """Ensure mpd is at the closest point when getting closer together
        """
        self.assertEqual(
            metrics.mpd(
                Position(np.array([0, 1]), np.array([1, 0]), 0),
                Position(np.array([1, 0]), np.array([-1, 0]), 0),
            ),
            1
        )

    def test_mpd_never_touch_getting_further_away(self):
        """Ensure mpd is at the closest point when getting further away
        """
        self.assertEqual(
            metrics.mpd(
                Position(np.array([0, 1]), np.array([-1, 0]), 0),
                Position(np.array([0, 0]), np.array([1, 0]), 0),
            ),
            1
        )

if __name__ == "__main__":
    unittest.main()
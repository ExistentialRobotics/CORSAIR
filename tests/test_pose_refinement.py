"""Behavioral checks for geometry-only acceptance and its optional overlap gate."""
import unittest
import numpy as np
from utils.pose_refinement import guard_pose_update


class PoseAcceptanceTests(unittest.TestCase):
    def test_lower_cost_can_be_accepted_despite_lower_overlap(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]])
        initial = np.eye(4)
        initial[1, 3] = .079
        proposal = np.eye(4)
        c, s = np.cos(.1), np.sin(.1)
        proposal[:2, :2] = [[c, -s], [s, c]]
        chosen, details = guard_pose_update(points, points, initial, proposal)
        self.assertTrue(details['accepted'])
        self.assertLess(details['proposed_overlap'], details['before_overlap'])
        np.testing.assert_array_equal(chosen, proposal)
        chosen, details = guard_pose_update(points, points, initial, proposal, require_overlap=True)
        self.assertFalse(details['accepted'])
        np.testing.assert_array_equal(chosen, initial)

    def test_equal_worse_and_invalid_proposals_retain_initial_pose(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]])
        initial = np.eye(4)
        worse = np.eye(4)
        worse[1, 3] = .1
        invalid = np.full((4, 4), np.nan)
        for proposal in [initial.copy(), worse, invalid]:
            with self.subTest(proposal=proposal):
                chosen, details = guard_pose_update(points, points, initial, proposal)
                self.assertFalse(details['accepted'])
                np.testing.assert_array_equal(chosen, initial)


if __name__ == '__main__':
    unittest.main()

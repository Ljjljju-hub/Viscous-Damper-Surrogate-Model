import unittest

import numpy as np

from mesh_motion import DamperMeshMotion, MeshRegion


class DamperMeshMotionTest(unittest.TestCase):
    def setUp(self):
        self.geometry = {
            "r1": 52.66,
            "sx": 54.30,
            "sy": 303.17,
            "a2": 45.69,
            "b1": 117.80,
            "b2": 91.85,
        }
        self.loading = {"A": 34.291, "Ts": 0.18}
        r = np.array([0.05266, 0.10696])
        z = np.array([0.11780, 0.269385, 0.42097, 0.51282, 0.664405, 0.81599])
        rr, zz = np.meshgrid(r, z, indexing="ij")
        self.reference_pos = np.column_stack((rr.ravel(), zz.ravel()))

    def test_three_region_weights_match_comsol_formulas(self):
        motion = DamperMeshMotion(
            self.reference_pos,
            self.geometry,
            self.loading,
        )
        expected_regions = [
            MeshRegion.LOWER,
            MeshRegion.LOWER,
            MeshRegion.MIDDLE,
            MeshRegion.MIDDLE,
            MeshRegion.UPPER,
            MeshRegion.UPPER,
        ]
        np.testing.assert_array_equal(motion.region, expected_regions * 2)
        np.testing.assert_allclose(
            motion.motion_weight,
            [0.0, 0.5, 1.0, 1.0, 0.5, 0.0] * 2,
            atol=1.0e-12,
        )

    def test_mesh_at_time_moves_only_in_z(self):
        motion = DamperMeshMotion(
            self.reference_pos,
            self.geometry,
            self.loading,
        )
        state = motion.at_time(self.loading["Ts"] / 2.0)
        expected_middle_displacement = 2.0 * self.loading["A"] * 1.0e-3

        np.testing.assert_allclose(state.pos[:, 0], self.reference_pos[:, 0])
        np.testing.assert_allclose(
            state.pos[:, 1] - self.reference_pos[:, 1],
            motion.motion_weight * expected_middle_displacement,
        )

    def test_mesh_velocity_is_analytical_time_derivative(self):
        motion = DamperMeshMotion(
            self.reference_pos,
            self.geometry,
            self.loading,
        )
        time = self.loading["Ts"] / 4.0
        state = motion.at_time(time)
        amplitude = self.loading["A"] * 1.0e-3
        expected_piston_velocity = amplitude * 2.0 * np.pi / self.loading["Ts"]

        np.testing.assert_allclose(state.mesh_velocity[:, 0], 0.0)
        np.testing.assert_allclose(
            state.mesh_velocity[:, 1],
            motion.motion_weight * expected_piston_velocity,
        )
        self.assertAlmostEqual(state.piston_velocity, expected_piston_velocity)


if __name__ == "__main__":
    unittest.main()

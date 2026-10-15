"""
test_spectral_braess.py
=======================
Automated verification tests for spectral properties and Braess' paradox.
"""

import unittest
import numpy as np
from spectral_braess_core import GridNetwork


class TestSpectralBraessCore(unittest.TestCase):

    def test_laplacian_properties(self):
        """Verify Laplacian L properties: row sums to 0, symmetric, PSD."""
        grid = GridNetwork(m=3, n=3, a=1.0)
        # Row sums of Laplacian should be 0
        np.testing.assert_allclose(np.sum(grid.L, axis=1), 0.0, atol=1e-10)
        # Symmetry
        np.testing.assert_allclose(grid.L, grid.L.T, atol=1e-10)
        # Eigenvalues >= 0 (positive semi-definite)
        eigvals = np.linalg.eigvalsh(grid.L)
        self.assertGreaterEqual(np.min(eigvals), -1e-10)

    def test_potential_monotonicity(self):
        """Lemma 2.7 from Lu & Miller: phi strictly decreases along every directed edge."""
        grid = GridNetwork(m=4, n=6, a=1.0)
        for (u, v) in grid.edges:
            phi_u = grid.phi_base[u]
            phi_v = grid.phi_base[v]
            self.assertGreater(phi_u, phi_v, f"Potential failed to decrease along edge {u}->{v}")

    def test_braess_chord_scarcity_on_squares(self):
        """As proved in Lu & Miller: square grids < 7x7 admit zero Braess chords."""
        grid_3x3 = GridNetwork(m=3, n=3)
        chords_3x3 = grid_3x3.find_all_chords()
        self.assertEqual(len(chords_3x3["braess_capable"]), 0)

        grid_5x5 = GridNetwork(m=5, n=5)
        chords_5x5 = grid_5x5.find_all_chords()
        self.assertEqual(len(chords_5x5["braess_capable"]), 0)

        grid_7x7 = GridNetwork(m=7, n=7)
        chords_7x7 = grid_7x7.find_all_chords()
        self.assertGreater(len(chords_7x7["braess_capable"]), 0)

    def test_braess_paradox_exact_cost_increase(self):
        """Theorem 2.15 verification: C_new - C_old = -V_{uv} * z* > 0."""
        grid = GridNetwork(m=3, n=6, a=1.0, q=1.0)
        chords = grid.find_all_chords()["braess_capable"]
        self.assertGreater(len(chords), 0)

        # Pick the first Braess chord
        u_coord, v_coord, V_uv, k = chords[0]
        self.assertLess(V_uv, 0.0)
        self.assertGreater(k, 0)

        u_idx = grid.node_to_idx[u_coord]
        v_idx = grid.node_to_idx[v_coord]
        R_uv = grid.effective_resistance(u_idx, v_idx)

        # Tune b according to Theorem 2.15: b = (-V_uv + eps) / k
        eps = 0.005
        b_val = (-V_uv + eps) / k
        grid.b = b_val

        f_base, cost_base = grid.solve_base_equilibrium()
        f_aug, cost_aug, z_star = grid.solve_augmented_equilibrium(u_coord, v_coord, c=0.0, d=0.0)

        self.assertGreater(z_star, 0.0)
        self.assertGreater(cost_aug, cost_base)

        diff = cost_aug - cost_base
        expected_diff = -V_uv * z_star
        np.testing.assert_allclose(diff, expected_diff, rtol=1e-3)
        print(f"\n[Test Passed] Verified Braess Paradox on 3x6: delta_Cost={diff:.6f} > 0, expected={expected_diff:.6f}")


if __name__ == "__main__":
    unittest.main()

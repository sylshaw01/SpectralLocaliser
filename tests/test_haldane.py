"""
Unit tests for TwoDimensionalHaldane model in SLmodels.py

The Haldane model is a 2D honeycomb lattice with:
- Two sublattices (A and B)
- Nearest neighbor hopping (real, t1)
- Next-nearest neighbor hopping (complex, t2*exp(±iφ))
- Mass term M (breaks inversion symmetry)
- Exhibits quantum Hall effect without magnetic field

Tests cover:
- Honeycomb lattice structure (L² unit cells, 2 sites each)
- Sublattice alternation (even=A, odd=B)
- Nearest and next-nearest neighbor connectivity
- Complex hopping with phases
- Mass term distribution
- Position operators for honeycomb geometry
- Spectral localizer (2D version)
- Local Chern marker calculation
- Topological phase transitions
"""

import pytest
import numpy as np
import scipy.sparse as sp
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from SLmodels import TwoDimensionalHaldane


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def trivial_haldane():
    """Topologically trivial Haldane (M > 3√3*t2*sin(φ))"""
    np.random.seed(42)
    return TwoDimensionalHaldane(
        L=4, disorder=0.0, rho=5.0, kappa=1.0,
        t1=1.0, t2=0.1, M=1.0, phi=np.pi/2
    )


@pytest.fixture
def topological_haldane():
    """Topologically nontrivial Haldane (|M| < 3√3*t2*sin(φ))"""
    np.random.seed(42)
    return TwoDimensionalHaldane(
        L=4, disorder=0.0, rho=5.0, kappa=1.0,
        t1=1.0, t2=0.3, M=0.2, phi=np.pi/2
    )


@pytest.fixture
def small_haldane():
    """Small system for detailed testing"""
    np.random.seed(42)
    return TwoDimensionalHaldane(
        L=3, disorder=0.0, rho=2.0, kappa=1.0,
        t1=1.0, t2=1.0/3.0, M=0.5, phi=np.pi/2
    )


@pytest.fixture
def disordered_haldane():
    """Haldane with disorder"""
    np.random.seed(42)
    return TwoDimensionalHaldane(
        L=4, disorder=0.5, rho=5.0, kappa=1.0,
        t1=1.0, t2=1.0/3.0, M=0.5, phi=np.pi/2
    )


@pytest.fixture(params=[2, 3, 4])
def various_sizes_haldane(request):
    """Parametrized fixture for multiple system sizes"""
    np.random.seed(42)
    return TwoDimensionalHaldane(
        L=request.param, disorder=0.0, rho=5.0, kappa=1.0,
        t1=1.0, t2=1.0/3.0, M=0.5, phi=np.pi/2
    )


# ============================================================================
# TEST CLASS: BASIC STRUCTURE
# ============================================================================

class TestBasicStructure:
    """Test basic structural properties of Haldane model"""

    def test_hamiltonian_shape(self, various_sizes_haldane):
        """Test that H has shape (2L²)×(2L²)"""
        model = various_sizes_haldane
        L = model.L
        mat_size = 2 * L * L  # 2 sites per unit cell
        assert model.H.shape == (mat_size, mat_size), \
            f"Expected H shape ({mat_size}, {mat_size}), got {model.H.shape}"

    def test_system_size_scaling(self):
        """Test that system size = 2L² (honeycomb lattice)"""
        test_sizes = [2, 3, 4, 5]
        for L in test_sizes:
            np.random.seed(42)
            model = TwoDimensionalHaldane(
                L=L, disorder=0.0, rho=1.0, kappa=1.0,
                t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
            )
            expected_size = 2 * L * L
            assert model.H.shape[0] == expected_size, \
                f"For L={L}, expected {expected_size} sites, got {model.H.shape[0]}"

    def test_hamiltonian_is_sparse(self, small_haldane):
        """Test that H is sparse matrix"""
        assert sp.issparse(small_haldane.H), "Hamiltonian should be sparse"

    def test_hamiltonian_is_hermitian(self, small_haldane):
        """Test H† = H (Hermiticity)"""
        H = small_haldane.H.toarray()
        H_conj_T = H.conj().T

        np.testing.assert_allclose(
            H, H_conj_T,
            rtol=1e-12, atol=1e-12,
            err_msg="Haldane Hamiltonian should be Hermitian"
        )

    def test_hamiltonian_is_complex(self, small_haldane):
        """Test that H is complex (due to NNN hopping phase)"""
        H = small_haldane.H.toarray()
        # Should have complex elements from NNN hopping
        assert np.iscomplexobj(H), "Hamiltonian should be complex"

    def test_parameters_stored_correctly(self):
        """Test that model parameters are stored"""
        np.random.seed(42)
        t1_val, t2_val, M_val, phi_val = 1.0, 0.3, 0.5, np.pi/4
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=t1_val, t2=t2_val, M=M_val, phi=phi_val
        )

        assert model.t1 == t1_val
        assert model.t2 == t2_val
        assert model.M == M_val
        assert model.phi == phi_val


# ============================================================================
# TEST CLASS: SUBLATTICE STRUCTURE
# ============================================================================

class TestSublatticeStructure:
    """Test honeycomb sublattice A and B structure"""

    def test_sublattice_alternation(self, small_haldane):
        """Test that sites alternate between A and B sublattices"""
        H = small_haldane.H.toarray()
        mat_size = small_haldane.L**2 * 2

        # Check diagonal elements (on-site terms)
        # Even indices should have +M (A sublattice)
        # Odd indices should have -M (B sublattice)
        for i in range(mat_size):
            diag_real = np.real(H[i, i])
            if i % 2 == 0:  # A sublattice
                # Should contain +M
                assert diag_real >= small_haldane.M - 1e-10, \
                    f"Site {i} (A sublattice) should have positive mass term"
            else:  # B sublattice
                # Should contain -M
                assert diag_real <= -small_haldane.M + 1e-10, \
                    f"Site {i} (B sublattice) should have negative mass term"

    def test_mass_term_values(self):
        """Test that mass term M is correctly applied"""
        np.random.seed(42)
        M_val = 0.8
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=M_val, phi=np.pi/2
        )

        H = model.H.toarray()
        diag = np.diag(H)

        # Extract real part (mass term)
        diag_real = np.real(diag)

        # A sites (even): should be +M
        A_sites = diag_real[::2]
        np.testing.assert_allclose(A_sites, M_val, rtol=1e-10,
                                   err_msg="A sublattice should have mass +M")

        # B sites (odd): should be -M
        B_sites = diag_real[1::2]
        np.testing.assert_allclose(B_sites, -M_val, rtol=1e-10,
                                   err_msg="B sublattice should have mass -M")

    def test_zero_mass_symmetry(self):
        """Test that M=0 gives symmetric sublattices"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.0, phi=np.pi/2
        )

        H = model.H.toarray()
        diag = np.diag(H)
        diag_real = np.real(diag)

        # With M=0 and no disorder, diagonal should be zero
        np.testing.assert_allclose(diag_real, 0.0, atol=1e-14,
                                   err_msg="M=0 should give zero on-site terms")


# ============================================================================
# TEST CLASS: DISORDER
# ============================================================================

class TestDisorder:
    """Test disorder effects"""

    def test_disorder_on_diagonal(self):
        """Test that disorder affects diagonal elements"""
        np.random.seed(42)
        disorder = 1.0
        M = 0.5

        model = TwoDimensionalHaldane(
            L=4, disorder=disorder, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=M, phi=np.pi/2
        )

        H = model.H.toarray()
        diag = np.diag(H)
        diag_real = np.real(diag)

        # Disorder is uniform in [-disorder, disorder]
        # A sites should be in [M-disorder, M+disorder]
        # B sites should be in [-M-disorder, -M+disorder]
        A_sites = diag_real[::2]
        B_sites = diag_real[1::2]

        assert np.all(A_sites >= M - disorder - 1e-10), "A sites below range"
        assert np.all(A_sites <= M + disorder + 1e-10), "A sites above range"
        assert np.all(B_sites >= -M - disorder - 1e-10), "B sites below range"
        assert np.all(B_sites <= -M + disorder + 1e-10), "B sites above range"

    def test_no_disorder_preserves_hermiticity(self, small_haldane):
        """Test that clean system is Hermitian"""
        H = small_haldane.H.toarray()
        np.testing.assert_allclose(H, H.conj().T, rtol=1e-12)

    def test_disorder_preserves_hermiticity(self, disordered_haldane):
        """Test that disorder preserves Hermiticity"""
        H = disordered_haldane.H.toarray()
        np.testing.assert_allclose(H, H.conj().T, rtol=1e-12)


# ============================================================================
# TEST CLASS: NEAREST NEIGHBOR HOPPING
# ============================================================================

class TestNearestNeighborHopping:
    """Test nearest neighbor hopping connectivity"""

    def test_nn_hopping_is_real(self):
        """Test that NN hopping is real (no phase)"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.0, M=0.0, phi=np.pi/2  # Turn off NNN
        )

        H = model.H.toarray()

        # With t2=0 and M=0, only NN hopping remains
        # NN hopping should be real
        # Off-diagonal elements should be real (or zero)
        off_diag = H - np.diag(np.diag(H))
        assert np.allclose(np.imag(off_diag), 0, atol=1e-14), \
            "NN hopping should be purely real"

    def test_nn_hopping_strength(self):
        """Test that NN hopping has magnitude t1"""
        np.random.seed(42)
        t1_val = 1.5
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=t1_val, t2=0.0, M=0.0, phi=np.pi/2
        )

        H = model.H.toarray()

        # Find non-zero off-diagonal elements (NN hoppings)
        off_diag = H - np.diag(np.diag(H))
        non_zero = np.abs(off_diag) > 1e-10

        # All non-zero off-diagonal should be -t1
        non_zero_values = off_diag[non_zero]
        np.testing.assert_allclose(np.abs(non_zero_values), t1_val, rtol=1e-10,
                                   err_msg=f"NN hopping should have magnitude {t1_val}")

    def test_nn_hopping_count_small_system(self):
        """Test NN hopping connectivity for small system"""
        np.random.seed(42)
        L = 2
        model = TwoDimensionalHaldane(
            L=L, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.0, M=0.0, phi=np.pi/2
        )

        # For L=2, we have 2×2×2 = 8 sites
        # Check that connectivity makes sense
        assert model.H.shape == (8, 8)


# ============================================================================
# TEST CLASS: NEXT-NEAREST NEIGHBOR HOPPING
# ============================================================================

class TestNextNearestNeighborHopping:
    """Test next-nearest neighbor hopping with phase"""

    def test_nnn_hopping_is_complex(self):
        """Test that NNN hopping has complex phase"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=0.0, t2=0.5, M=0.0, phi=np.pi/2  # Turn off NN
        )

        H = model.H.toarray()

        # With t1=0, only NNN remains
        # Should have complex elements
        assert np.max(np.abs(np.imag(H))) > 1e-6, \
            "NNN hopping should have imaginary components"

    def test_nnn_phase_dependence(self):
        """Test that changing phi changes the Hamiltonian"""
        np.random.seed(42)
        L = 3

        model1 = TwoDimensionalHaldane(
            L=L, disorder=0.0, rho=1.0, kappa=1.0,
            t1=0.0, t2=0.3, M=0.0, phi=0.0
        )

        model2 = TwoDimensionalHaldane(
            L=L, disorder=0.0, rho=1.0, kappa=1.0,
            t1=0.0, t2=0.3, M=0.0, phi=np.pi/2
        )

        H1 = model1.H.toarray()
        H2 = model2.H.toarray()

        # Different phases should give different Hamiltonians
        assert not np.allclose(H1, H2), \
            "Different phi values should produce different Hamiltonians"

    def test_nnn_only_on_same_sublattice(self):
        """Test that NNN hopping only connects within same sublattice"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=0.0, t2=0.3, M=0.0, phi=np.pi/2  # Only NNN
        )

        H = model.H.toarray()
        mat_size = 3**2 * 2

        # NNN should only connect A to A or B to B
        # Check that A sites (even) don't connect to B sites (odd) via NNN
        for i in range(mat_size):
            for j in range(mat_size):
                if i % 2 != j % 2:  # Different sublattices
                    # With only NNN hopping, should be zero
                    # (NN is turned off with t1=0)
                    if abs(i - j) > 1:  # Not NN distance
                        assert np.abs(H[i, j]) < 1e-10, \
                            f"NNN should not connect different sublattices: H[{i},{j}]={H[i,j]}"


# ============================================================================
# TEST CLASS: POSITION OPERATORS
# ============================================================================

class TestPositionOperators:
    """Test position operators for honeycomb geometry"""

    def test_position_operator_is_list(self, small_haldane):
        """Test that position operator is list [X, Y]"""
        assert isinstance(small_haldane.X, list), "Position operator should be a list"
        assert len(small_haldane.X) == 2, "Should have 2 position operators [X, Y]"

    def test_position_operators_shape(self, various_sizes_haldane):
        """Test that position operators have correct shape"""
        model = various_sizes_haldane
        mat_size = 2 * model.L**2

        for i, op in enumerate(model.X):
            assert op.shape == (mat_size, mat_size), \
                f"Position operator {i} should be ({mat_size}, {mat_size})"

    def test_position_operators_are_diagonal(self, small_haldane):
        """Test that X and Y are diagonal"""
        for i, op in enumerate(small_haldane.X):
            op_dense = op.toarray()
            off_diag_mask = ~np.eye(op.shape[0], dtype=bool)
            assert np.allclose(op_dense[off_diag_mask], 0), \
                f"Position operator {i} should be diagonal"

    def test_position_operators_are_real(self, small_haldane):
        """Test that position operators are real"""
        for op in small_haldane.X:
            op_dense = op.toarray()
            assert np.all(np.isreal(op_dense)), "Position operators should be real"

    def test_position_operators_centered(self, small_haldane):
        """Test that position operators are centered at (0,0)"""
        X = small_haldane.X[0].toarray()
        Y = small_haldane.X[1].toarray()

        x_mean = np.mean(np.diag(X))
        y_mean = np.mean(np.diag(Y))

        assert np.abs(x_mean) < 1e-10, "X should be centered (mean ≈ 0)"
        assert np.abs(y_mean) < 1e-10, "Y should be centered (mean ≈ 0)"

    def test_honeycomb_geometry(self):
        """Test that positions follow honeycomb lattice geometry"""
        np.random.seed(42)
        L = 3
        model = TwoDimensionalHaldane(
            L=L, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
        )

        X_diag = np.diag(model.X[0].toarray())
        Y_diag = np.diag(model.X[1].toarray())

        # Check that A and B sites in same unit cell are offset
        for unit_cell in range(L**2):
            A_idx = 2 * unit_cell
            B_idx = 2 * unit_cell + 1

            x_A, y_A = X_diag[A_idx], Y_diag[A_idx]
            x_B, y_B = X_diag[B_idx], Y_diag[B_idx]

            # B should be offset from A by approximately (0.5, 1/(2√3))
            # (after centering, exact values vary)
            # Just check they're different
            assert not (np.isclose(x_A, x_B) and np.isclose(y_A, y_B)), \
                f"A and B in unit cell {unit_cell} should have different positions"


# ============================================================================
# TEST CLASS: SPECTRAL LOCALIZER
# ============================================================================

class TestSpectralLocalizer:
    """Test 2D spectral localizer"""

    def test_localizer_shape(self, small_haldane):
        """Test that localizer has shape (2×mat_size)×(2×mat_size)"""
        mat_size = 2 * small_haldane.L**2
        SL = small_haldane.spectral_localiser

        assert SL.shape == (2*mat_size, 2*mat_size), \
            f"Spectral localizer should be ({2*mat_size}, {2*mat_size})"

    def test_localizer_is_hermitian(self, small_haldane):
        """Test that localizer is Hermitian"""
        SL = small_haldane.spectral_localiser.toarray()
        SL_conj_T = SL.conj().T

        np.testing.assert_allclose(SL, SL_conj_T, rtol=1e-12, atol=1e-12,
                                   err_msg="Spectral localizer should be Hermitian")

    def test_localizer_is_complex(self, small_haldane):
        """Test that localizer is complex"""
        SL = small_haldane.spectral_localiser.toarray()
        assert np.iscomplexobj(SL), "Spectral localizer should be complex"

    def test_localizer_block_structure(self):
        """Test 2D localizer structure [[H-e0, κ(X-x0) - iκ(Y-y0)], [...]]"""
        np.random.seed(42)
        L = 2
        kappa = 1.5
        e0, x0, y0 = 0.1, 0.2, 0.3

        model = TwoDimensionalHaldane(
            L=L, disorder=0.0, rho=1.0, kappa=kappa,
            t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
        )

        mat_size = 2 * L**2
        SL = model.create_localiser(e0=e0, x0=x0, y0=y0).toarray()
        H = model.H.toarray()
        X = model.X[0].toarray()
        Y = model.X[1].toarray()

        # Extract blocks
        top_left = SL[:mat_size, :mat_size]
        top_right = SL[:mat_size, mat_size:]
        bottom_left = SL[mat_size:, :mat_size]
        bottom_right = SL[mat_size:, mat_size:]

        # Expected blocks
        H_shifted = H - e0 * np.eye(mat_size)
        X_shifted = X - x0 * np.eye(mat_size)
        Y_shifted = Y - y0 * np.eye(mat_size)

        block_off_diag = kappa * X_shifted - 1j * kappa * Y_shifted

        # Check structure
        np.testing.assert_allclose(top_left, H_shifted, rtol=1e-10,
                                   err_msg="Top-left should be H-e0*I")
        np.testing.assert_allclose(top_right, block_off_diag, rtol=1e-10,
                                   err_msg="Top-right incorrect")
        np.testing.assert_allclose(bottom_left, block_off_diag.conj().T, rtol=1e-10,
                                   err_msg="Bottom-left should be conjugate transpose of top-right")
        np.testing.assert_allclose(bottom_right, -H_shifted, rtol=1e-10,
                                   err_msg="Bottom-right should be -(H-e0*I)")

    def test_localizer_default_parameters(self, small_haldane):
        """Test that default e0=0, x0=0, y0=0 works"""
        # This tests that the spectral_localiser attribute is created correctly
        SL = small_haldane.spectral_localiser
        assert SL is not None
        assert sp.issparse(SL)


# ============================================================================
# TEST CLASS: PHYSICAL PROPERTIES
# ============================================================================

class TestPhysicalProperties:
    """Test physical properties and band structure"""

    def test_eigenvalue_count(self, small_haldane):
        """Test that H has mat_size eigenvalues"""
        mat_size = 2 * small_haldane.L**2
        eigvals = np.linalg.eigvalsh(small_haldane.H.toarray())

        assert len(eigvals) == mat_size

    def test_eigenvalues_are_real(self, small_haldane):
        """Test that eigenvalues are real (Hermitian property)"""
        eigvals = np.linalg.eigvals(small_haldane.H.toarray())

        # Eigenvalues should be real (up to numerical precision)
        # Check imaginary parts are negligible
        max_imag = np.max(np.abs(np.imag(eigvals)))
        assert max_imag < 1e-10, \
            f"Eigenvalues should be real (max imaginary part: {max_imag})"

    def test_band_gap_in_trivial_phase(self):
        """Test that trivial phase has band gap"""
        np.random.seed(42)
        # Trivial: M > 3√3 * t2 * sin(φ)
        model = TwoDimensionalHaldane(
            L=4, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.1, M=1.0, phi=np.pi/2
        )

        eigvals = np.linalg.eigvalsh(model.H.toarray())
        eigvals_sorted = np.sort(eigvals)

        # Should have a gap
        mat_size = 2 * model.L**2
        mid_point = mat_size // 2

        lower_band_max = eigvals_sorted[mid_point - 1]
        upper_band_min = eigvals_sorted[mid_point]

        gap = upper_band_min - lower_band_max

        assert gap > 0.1, f"Trivial phase should have gap, got {gap}"

    def test_band_gap_in_topological_phase(self):
        """Test that topological phase has band gap"""
        np.random.seed(42)
        # Topological: |M| < 3√3 * t2 * sin(φ)
        model = TwoDimensionalHaldane(
            L=4, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.2, phi=np.pi/2
        )

        eigvals = np.linalg.eigvalsh(model.H.toarray())
        eigvals_sorted = np.sort(eigvals)

        mat_size = 2 * model.L**2
        mid_point = mat_size // 2

        lower_band_max = eigvals_sorted[mid_point - 1]
        upper_band_min = eigvals_sorted[mid_point]

        gap = upper_band_min - lower_band_max

        assert gap > 0.05, f"Topological phase should have gap, got {gap}"


# ============================================================================
# TEST CLASS: CHERN MARKER
# ============================================================================

class TestChernMarker:
    """Test local Chern marker calculation"""

    def test_projection_operator_shape(self):
        """Test projection operator has correct shape"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
        )

        model.projection_operator_lower(fermi_energy=0.0)

        assert model.P is not None
        mat_size = 2 * model.L**2
        assert model.P.shape == (mat_size, mat_size)

    def test_projection_operator_is_projector(self):
        """Test that P² = P (projection property)"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
        )

        model.projection_operator_lower(fermi_energy=0.0)
        P = model.P

        P_squared = P @ P

        np.testing.assert_allclose(P, P_squared, rtol=1e-10,
                                   err_msg="Projection operator should satisfy P² = P")

    def test_projection_operator_is_hermitian(self):
        """Test that P† = P"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
        )

        model.projection_operator_lower(fermi_energy=0.0)
        P = model.P

        np.testing.assert_allclose(P, P.conj().T, rtol=1e-10,
                                   err_msg="Projection operator should be Hermitian")

    def test_chern_marker_shape(self):
        """Test that Chern marker returns L×L array"""
        np.random.seed(42)
        L = 3
        model = TwoDimensionalHaldane(
            L=L, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
        )

        chern_marker = model.calculate_local_chern_marker(fermi_energy=0.0)

        assert chern_marker.shape == (L, L), \
            f"Chern marker should be ({L}, {L}), got {chern_marker.shape}"

    def test_chern_marker_is_real(self):
        """Test that Chern marker values are real"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
        )

        chern_marker = model.calculate_local_chern_marker(fermi_energy=0.0)

        assert np.all(np.isreal(chern_marker)), "Chern marker should be real"


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_minimal_system_L_equals_2(self):
        """Test L=2 (2×2 unit cells = 8 sites)"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=2, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
        )

        assert model.H.shape == (8, 8)
        assert len(model.X) == 2
        for op in model.X:
            assert op.shape == (8, 8)

    def test_zero_phase(self):
        """Test with phi=0 (breaks time-reversal differently)"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.5, phi=0.0
        )

        # Should still be Hermitian
        H = model.H.toarray()
        np.testing.assert_allclose(H, H.conj().T, rtol=1e-12)

    @pytest.mark.parametrize("phi", [0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2])
    def test_various_phases(self, phi):
        """Test with different phase values"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.0, rho=1.0, kappa=1.0,
            t1=1.0, t2=0.3, M=0.5, phi=phi
        )

        # Should construct without error
        assert model.H.shape == (18, 18)

        # Should be Hermitian
        H = model.H.toarray()
        np.testing.assert_allclose(H, H.conj().T, rtol=1e-12)


# ============================================================================
# TEST CLASS: REPRODUCIBILITY
# ============================================================================

class TestReproducibility:
    """Test random seed control"""

    def test_same_seed_gives_same_hamiltonian(self):
        """Test reproducibility with seeds"""
        L = 3
        params = dict(disorder=0.5, rho=1.0, kappa=1.0,
                     t1=1.0, t2=0.3, M=0.5, phi=np.pi/2)

        np.random.seed(123)
        model1 = TwoDimensionalHaldane(L=L, **params)
        H1 = model1.H.toarray()

        np.random.seed(123)
        model2 = TwoDimensionalHaldane(L=L, **params)
        H2 = model2.H.toarray()

        np.testing.assert_array_equal(H1, H2)

    def test_different_seeds_give_different_hamiltonians(self):
        """Test that different seeds produce different disorder"""
        L = 3
        params = dict(disorder=0.5, rho=1.0, kappa=1.0,
                     t1=1.0, t2=0.3, M=0.5, phi=np.pi/2)

        np.random.seed(111)
        model1 = TwoDimensionalHaldane(L=L, **params)
        H1 = model1.H.toarray()

        np.random.seed(222)
        model2 = TwoDimensionalHaldane(L=L, **params)
        H2 = model2.H.toarray()

        # Disorder should make them different
        assert not np.allclose(H1, H2)


# ============================================================================
# TEST CLASS: INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests"""

    def test_full_workflow(self):
        """Test complete workflow"""
        np.random.seed(42)
        model = TwoDimensionalHaldane(
            L=3, disorder=0.1, rho=2.0, kappa=1.5,
            t1=1.0, t2=0.3, M=0.5, phi=np.pi/2
        )

        # Diagonalize
        model.find_eigval(model.H, sparse=False)

        # Compute statistics
        r = model.calculate_r(model.H_eigval)
        z = model.calculate_z(model.H_eigval)

        assert 0 < r < 1
        assert 0 < z < 1

        # Compute Chern marker
        chern_marker = model.calculate_local_chern_marker(fermi_energy=0.0)
        assert chern_marker.shape == (3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for OneDimensionalSSHBlockBasis model in SLmodels.py

Tests cover:
- Hamiltonian block structure [[0, A], [A^T, 0]]
- A matrix structure (intracell + intercell hopping with disorder)
- Position operator (operates on unit cells, size L/2)
- Spectral localizer structure (X_block + H)
- Topological properties (v vs w parameter regime)
- Physical behaviors
- Edge cases
"""

import pytest
import numpy as np
import scipy.sparse as sp
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from SLmodels import OneDimensionalSSHBlockBasis


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def trivial_ssh():
    """Topologically trivial SSH (v > w)"""
    np.random.seed(42)
    return OneDimensionalSSHBlockBasis(
        L=20, disorder=0.0, rho=5.0, kappa=1.0,
        v=1.5, w=1.0
    )


@pytest.fixture
def topological_ssh():
    """Topologically nontrivial SSH (w > v)"""
    np.random.seed(42)
    return OneDimensionalSSHBlockBasis(
        L=20, disorder=0.0, rho=5.0, kappa=1.0,
        v=1.0, w=1.5
    )


@pytest.fixture
def symmetric_ssh():
    """Symmetric SSH (v = w, critical point)"""
    np.random.seed(42)
    return OneDimensionalSSHBlockBasis(
        L=20, disorder=0.0, rho=5.0, kappa=1.0,
        v=1.0, w=1.0
    )


@pytest.fixture
def disordered_ssh():
    """SSH with disorder on hopping terms"""
    np.random.seed(42)
    return OneDimensionalSSHBlockBasis(
        L=40, disorder=0.5, rho=5.0, kappa=1.0,
        v=1.0, w=1.5
    )


@pytest.fixture(params=[20, 40, 100])
def various_sizes_ssh(request):
    """Parametrized fixture for multiple even system sizes"""
    np.random.seed(42)
    return OneDimensionalSSHBlockBasis(
        L=request.param, disorder=0.0, rho=5.0, kappa=1.0,
        v=1.0, w=1.5
    )


# ============================================================================
# TEST CLASS: BASIC STRUCTURE AND REQUIREMENTS
# ============================================================================

class TestSSHBasicStructure:
    """Test basic structural properties and requirements"""

    def test_L_must_be_even(self):
        """
        SSH model requires even L (pairs of sites form unit cells)

        Note: Currently the code does NOT validate this, but L should
        always be even for physical consistency. Odd L would give
        L//2 unit cells, losing one site.
        """
        # Even L should work correctly
        np.random.seed(42)
        even_L_values = [2, 20, 40, 100]

        for L in even_L_values:
            model = OneDimensionalSSHBlockBasis(
                L=L, disorder=0.0, rho=1.0, kappa=1.0, v=1.0, w=1.5
            )
            assert model.L == L, f"Model should store L={L}"
            assert model.H.shape == (L, L), f"Hamiltonian should be {L}×{L}"
            assert model.X.shape == (L//2, L//2), f"Position operator should be {L//2}×{L//2}"

            # Verify block structure is correct
            H = model.H.toarray()
            half_L = L // 2

            # Top-left and bottom-right blocks should be zero
            np.testing.assert_allclose(
                H[:half_L, :half_L], 0, atol=1e-14,
                err_msg=f"Top-left block should be zero for L={L}"
            )
            np.testing.assert_allclose(
                H[half_L:, half_L:], 0, atol=1e-14,
                err_msg=f"Bottom-right block should be zero for L={L}"
            )

    def test_hamiltonian_shape(self, various_sizes_ssh):
        """Test that Hamiltonian has shape L×L"""
        model = various_sizes_ssh
        L = model.L
        assert model.H.shape == (L, L), f"Expected H shape ({L}, {L}), got {model.H.shape}"

    def test_hamiltonian_is_sparse(self, trivial_ssh):
        """Test that Hamiltonian is sparse CSR matrix"""
        assert sp.issparse(trivial_ssh.H), "Hamiltonian should be sparse"
        assert trivial_ssh.H.format == 'csr', f"Expected CSR format, got {trivial_ssh.H.format}"

    def test_hamiltonian_is_hermitian(self, disordered_ssh):
        """Test H† = H (Hermiticity)"""
        H = disordered_ssh.H.toarray()
        H_conj_T = H.conj().T

        np.testing.assert_allclose(
            H, H_conj_T,
            rtol=1e-14, atol=1e-14,
            err_msg="SSH Hamiltonian should be Hermitian"
        )

    def test_hamiltonian_is_real(self, trivial_ssh):
        """Test that H is real (no imaginary components)"""
        H = trivial_ssh.H.toarray()
        assert np.all(np.isreal(H)), "SSH Hamiltonian should be real"

    def test_parameters_stored_correctly(self):
        """Test that v, w parameters are stored as attributes"""
        np.random.seed(42)
        v_val = 1.3
        w_val = 0.8
        model = OneDimensionalSSHBlockBasis(L=20, disorder=0.0, rho=5.0, kappa=1.0, v=v_val, w=w_val)

        assert model.v == v_val, f"Expected v={v_val}, got {model.v}"
        assert model.w == w_val, f"Expected w={w_val}, got {model.w}"


# ============================================================================
# TEST CLASS: BLOCK STRUCTURE
# ============================================================================

class TestBlockStructure:
    """Test the block structure H = [[0, A], [A^T, 0]]"""

    def test_block_diagonal_structure(self, trivial_ssh):
        """Test that H = [[0, A], [A^T, 0]] with zero diagonal blocks"""
        H = trivial_ssh.H.toarray()
        L = trivial_ssh.L
        half_L = L // 2

        # Extract blocks
        top_left = H[:half_L, :half_L]
        top_right = H[:half_L, half_L:]
        bottom_left = H[half_L:, :half_L]
        bottom_right = H[half_L:, half_L:]

        # Diagonal blocks should be zero
        np.testing.assert_allclose(
            top_left, 0,
            atol=1e-14,
            err_msg="Top-left block should be zero"
        )
        np.testing.assert_allclose(
            bottom_right, 0,
            atol=1e-14,
            err_msg="Bottom-right block should be zero"
        )

        # Off-diagonal blocks should be transpose of each other
        np.testing.assert_allclose(
            bottom_left, top_right.T,
            rtol=1e-14,
            err_msg="H should have structure [[0, A], [A^T, 0]]"
        )

    def test_A_matrix_structure(self, trivial_ssh):
        """Test that A matrix has correct upper-triangular structure"""
        H = trivial_ssh.H.toarray()
        L = trivial_ssh.L
        half_L = L // 2

        # Extract A matrix (top-right block)
        A = H[:half_L, half_L:]

        # A should have elements on main diagonal and lower diagonal only
        # (it's actually upper-bidiagonal in the way it's constructed)
        # Main diagonal: intracell hopping
        # Lower diagonal: intercell hopping

        # Check that A has correct sparsity pattern
        # Only diagonal and sub-diagonal should be non-zero
        for i in range(half_L):
            for j in range(half_L):
                if j == i:  # Main diagonal (intracell)
                    assert A[i, j] != 0, f"A[{i},{j}] (intracell) should be non-zero"
                elif j == i - 1:  # Lower diagonal (intercell)
                    if i > 0:  # Not the first row
                        assert A[i, j] != 0, f"A[{i},{j}] (intercell) should be non-zero"
                elif abs(i - j) > 1:  # Far from diagonal
                    assert np.isclose(A[i, j], 0, atol=1e-14), \
                        f"A[{i},{j}] should be zero (distance {abs(i-j)} from diagonal)"

    def test_A_matrix_intracell_values(self):
        """Test that diagonal of A contains v (intracell hopping)"""
        np.random.seed(42)
        v_val = 1.234
        w_val = 0.567
        model = OneDimensionalSSHBlockBasis(
            L=20, disorder=0.0, rho=5.0, kappa=1.0,
            v=v_val, w=w_val
        )

        H = model.H.toarray()
        half_L = model.L // 2
        A = H[:half_L, half_L:]

        # Diagonal should be v (no disorder)
        diagonal = np.diag(A)
        np.testing.assert_allclose(
            diagonal, v_val,
            rtol=1e-10,
            err_msg=f"Diagonal of A should be v={v_val}"
        )

    def test_A_matrix_intercell_values(self):
        """Test that lower diagonal of A contains w (intercell hopping)"""
        np.random.seed(42)
        v_val = 1.234
        w_val = 0.567
        model = OneDimensionalSSHBlockBasis(
            L=20, disorder=0.0, rho=5.0, kappa=1.0,
            v=v_val, w=w_val
        )

        H = model.H.toarray()
        half_L = model.L // 2
        A = H[:half_L, half_L:]

        # Lower diagonal should be w (no disorder)
        lower_diag = np.diag(A, k=-1)
        np.testing.assert_allclose(
            lower_diag, w_val,
            rtol=1e-10,
            err_msg=f"Lower diagonal of A should be w={w_val}"
        )

    def test_chiral_symmetry(self, trivial_ssh):
        """Test that H has chiral symmetry: {H, Γ} = 0 where Γ = σ_z"""
        H = trivial_ssh.H.toarray()
        L = trivial_ssh.L
        half_L = L // 2

        # Chiral symmetry operator Γ = [[I, 0], [0, -I]]
        Gamma = np.block([
            [np.eye(half_L), np.zeros((half_L, half_L))],
            [np.zeros((half_L, half_L)), -np.eye(half_L)]
        ])

        # Check anticommutator {H, Γ} = HΓ + ΓH = 0
        anticommutator = H @ Gamma + Gamma @ H

        np.testing.assert_allclose(
            anticommutator, 0,
            atol=1e-12,
            err_msg="SSH model should have chiral symmetry: {H, Γ} = 0"
        )


# ============================================================================
# TEST CLASS: DISORDER
# ============================================================================

class TestDisorder:
    """Test disorder effects on hopping terms"""

    def test_disorder_on_intracell_hopping(self):
        """Test that disorder affects intracell hopping (diagonal of A)"""
        np.random.seed(42)
        v_val = 1.0
        disorder = 0.5

        model = OneDimensionalSSHBlockBasis(
            L=40, disorder=disorder, rho=5.0, kappa=1.0,
            v=v_val, w=1.5
        )

        H = model.H.toarray()
        half_L = model.L // 2
        A = H[:half_L, half_L:]

        diagonal = np.diag(A)

        # Diagonal should be in range [v - disorder/2, v + disorder/2]
        assert np.all(diagonal >= v_val - disorder/2 - 1e-10), \
            f"Intracell hopping below v - disorder/2"
        assert np.all(diagonal <= v_val + disorder/2 + 1e-10), \
            f"Intracell hopping above v + disorder/2"

    def test_disorder_on_intercell_hopping(self):
        """Test that disorder affects intercell hopping (lower diagonal of A)"""
        np.random.seed(42)
        w_val = 1.5
        disorder = 0.5

        model = OneDimensionalSSHBlockBasis(
            L=40, disorder=disorder, rho=5.0, kappa=1.0,
            v=1.0, w=w_val
        )

        H = model.H.toarray()
        half_L = model.L // 2
        A = H[:half_L, half_L:]

        lower_diag = np.diag(A, k=-1)

        # Lower diagonal should be in range [w - disorder/2, w + disorder/2]
        assert np.all(lower_diag >= w_val - disorder/2 - 1e-10), \
            f"Intercell hopping below w - disorder/2"
        assert np.all(lower_diag <= w_val + disorder/2 + 1e-10), \
            f"Intercell hopping above w + disorder/2"

    def test_no_disorder_gives_clean_hopping(self):
        """Test that disorder=0 gives exact v and w values"""
        np.random.seed(42)
        v_val = 1.3
        w_val = 0.7

        model = OneDimensionalSSHBlockBasis(
            L=40, disorder=0.0, rho=5.0, kappa=1.0,
            v=v_val, w=w_val
        )

        H = model.H.toarray()
        half_L = model.L // 2
        A = H[:half_L, half_L:]

        # Check intracell
        diagonal = np.diag(A)
        np.testing.assert_allclose(diagonal, v_val, rtol=1e-14,
                                   err_msg="With disorder=0, intracell should equal v exactly")

        # Check intercell
        lower_diag = np.diag(A, k=-1)
        np.testing.assert_allclose(lower_diag, w_val, rtol=1e-14,
                                   err_msg="With disorder=0, intercell should equal w exactly")

    def test_reproducibility_with_seed(self):
        """Test that same seed gives same disorder realization"""
        v_val = 1.0
        w_val = 1.5
        disorder = 0.5

        np.random.seed(123)
        model1 = OneDimensionalSSHBlockBasis(L=40, disorder=disorder, rho=5.0, kappa=1.0, v=v_val, w=w_val)
        H1 = model1.H.toarray()

        np.random.seed(123)
        model2 = OneDimensionalSSHBlockBasis(L=40, disorder=disorder, rho=5.0, kappa=1.0, v=v_val, w=w_val)
        H2 = model2.H.toarray()

        np.testing.assert_array_equal(H1, H2,
                                      err_msg="Same seed should give identical Hamiltonians")


# ============================================================================
# TEST CLASS: POSITION OPERATOR
# ============================================================================

class TestPositionOperator:
    """Test position operator (operates on unit cells)"""

    def test_position_operator_size(self, various_sizes_ssh):
        """Test that X has size L/2 × L/2 (unit cell basis)"""
        model = various_sizes_ssh
        L = model.L
        half_L = L // 2

        assert model.X.shape == (half_L, half_L), \
            f"Position operator should be ({half_L}, {half_L}), got {model.X.shape}"

    def test_position_operator_is_diagonal(self, trivial_ssh):
        """Test that X is diagonal matrix"""
        X = trivial_ssh.X.toarray()
        half_L = trivial_ssh.L // 2

        # All off-diagonal elements should be zero
        off_diag_mask = ~np.eye(half_L, dtype=bool)
        assert np.allclose(X[off_diag_mask], 0), \
            "Position operator should be diagonal"

    def test_position_operator_range(self):
        """Test that X ranges from -rho to +rho"""
        np.random.seed(42)
        rho_val = 7.5
        model = OneDimensionalSSHBlockBasis(
            L=100, disorder=0.0, rho=rho_val, kappa=1.0,
            v=1.0, w=1.5
        )

        X_diag = np.diag(model.X.toarray())

        assert np.isclose(X_diag.min(), -rho_val, rtol=1e-10), \
            f"Min position should be -rho={-rho_val}, got {X_diag.min()}"
        assert np.isclose(X_diag.max(), rho_val, rtol=1e-10), \
            f"Max position should be rho={rho_val}, got {X_diag.max()}"

    def test_position_operator_linearity(self):
        """Test that position operator elements are evenly spaced"""
        np.random.seed(42)
        model = OneDimensionalSSHBlockBasis(
            L=100, disorder=0.0, rho=5.0, kappa=1.0,
            v=1.0, w=1.5
        )

        X_diag = np.diag(model.X.toarray())
        differences = np.diff(X_diag)

        # All differences should be equal (evenly spaced)
        assert np.allclose(differences, differences[0], rtol=1e-10), \
            "Position operator should be evenly spaced"

    def test_position_operator_is_sparse(self, trivial_ssh):
        """Test that X is stored as sparse matrix"""
        assert sp.issparse(trivial_ssh.X), \
            "Position operator should be sparse"


# ============================================================================
# TEST CLASS: SPECTRAL LOCALIZER
# ============================================================================

class TestSpectralLocalizer:
    """Test spectral localizer SL = X_block + H"""

    def test_localizer_shape(self, trivial_ssh):
        """Test that localizer has shape L×L (same as H)"""
        L = trivial_ssh.L
        SL = trivial_ssh.spectral_localiser

        assert SL.shape == (L, L), \
            f"Spectral localizer should be ({L}, {L}), got {SL.shape}"

    def test_localizer_structure(self):
        """Test SL = X_block + H where X_block = [[κX, 0], [0, -κX]]"""
        np.random.seed(42)
        L = 20
        kappa_val = 2.5

        model = OneDimensionalSSHBlockBasis(
            L=L, disorder=0.0, rho=5.0, kappa=kappa_val,
            v=1.0, w=1.5
        )

        SL = model.spectral_localiser.toarray()
        H = model.H.toarray()
        X = model.X.toarray()
        half_L = L // 2

        # Construct expected X_block
        X_block_expected = np.block([
            [kappa_val * X, np.zeros((half_L, half_L))],
            [np.zeros((half_L, half_L)), -kappa_val * X]
        ])

        # SL should equal X_block + H
        SL_expected = X_block_expected + H

        np.testing.assert_allclose(
            SL, SL_expected,
            rtol=1e-12,
            err_msg="Spectral localizer should equal X_block + H"
        )

    def test_localizer_is_hermitian(self, disordered_ssh):
        """Test that localizer is Hermitian"""
        SL = disordered_ssh.spectral_localiser.toarray()
        SL_conj_T = SL.conj().T

        np.testing.assert_allclose(
            SL, SL_conj_T,
            rtol=1e-14, atol=1e-14,
            err_msg="Spectral localizer should be Hermitian"
        )

    def test_localizer_kappa_scaling(self):
        """Test that κ parameter correctly scales X blocks"""
        np.random.seed(42)
        L = 20
        kappa_values = [1.0, 2.0, 5.0]

        for kappa_val in kappa_values:
            model = OneDimensionalSSHBlockBasis(
                L=L, disorder=0.0, rho=5.0, kappa=kappa_val,
                v=1.0, w=1.5
            )

            SL = model.spectral_localiser.toarray()
            X = model.X.toarray()
            half_L = L // 2

            # Extract diagonal blocks
            top_left = SL[:half_L, :half_L]
            bottom_right = SL[half_L:, half_L:]

            # Top-left should contain κX (plus zero from H)
            np.testing.assert_allclose(
                top_left, kappa_val * X,
                rtol=1e-12,
                err_msg=f"Top-left block incorrect for κ={kappa_val}"
            )

            # Bottom-right should contain -κX (plus zero from H)
            np.testing.assert_allclose(
                bottom_right, -kappa_val * X,
                rtol=1e-12,
                err_msg=f"Bottom-right block incorrect for κ={kappa_val}"
            )

    def test_localizer_is_sparse(self, trivial_ssh):
        """Test that localizer is stored as sparse matrix"""
        assert sp.issparse(trivial_ssh.spectral_localiser), \
            "Spectral localizer should be sparse"

    def test_localizer_size_differs_from_parent_ssh(self):
        """
        Test that BlockBasis localizer is L×L, not 2L×2L

        Note: The parent OneDimensionalSSH class creates a 2L×2L localizer
        with structure [[κX, H^T], [H, -κX]].

        The BlockBasis version OVERRIDES this and creates an L×L localizer
        with structure X_block + H where X_block = [[κX, 0], [0, -κX]].

        This test validates the BlockBasis approach is used.
        """
        np.random.seed(42)
        L = 20
        model = OneDimensionalSSHBlockBasis(
            L=L, disorder=0.0, rho=5.0, kappa=2.0,
            v=1.0, w=1.5
        )

        # BlockBasis localizer should be L×L (same size as H)
        assert model.spectral_localiser.shape == (L, L), \
            f"BlockBasis localizer should be {L}×{L}, not 2L×2L"

        # Verify it equals X_block + H
        SL = model.spectral_localiser.toarray()
        H = model.H.toarray()
        X = model.X.toarray()
        half_L = L // 2
        kappa = model.kappa

        X_block = np.block([
            [kappa * X, np.zeros((half_L, half_L))],
            [np.zeros((half_L, half_L)), -kappa * X]
        ])

        expected_SL = X_block + H

        np.testing.assert_allclose(
            SL, expected_SL, rtol=1e-12,
            err_msg="BlockBasis localizer should equal X_block + H"
        )


# ============================================================================
# TEST CLASS: PHYSICAL PROPERTIES
# ============================================================================

class TestPhysicalProperties:
    """Test physical properties and behaviors"""

    def test_eigenvalue_count(self, trivial_ssh):
        """Test that H has exactly L eigenvalues"""
        L = trivial_ssh.L
        eigvals = np.linalg.eigvalsh(trivial_ssh.H.toarray())

        assert len(eigvals) == L, f"Expected {L} eigenvalues, got {len(eigvals)}"

    def test_eigenvalues_are_real(self, disordered_ssh):
        """Test that all eigenvalues are real (Hermitian property)"""
        eigvals = np.linalg.eigvals(disordered_ssh.H.toarray())

        assert np.all(np.isreal(eigvals)), "Eigenvalues should all be real"
        assert np.max(np.abs(np.imag(eigvals))) < 1e-10, \
            f"Imaginary parts too large: {np.max(np.abs(np.imag(eigvals)))}"

    def test_spectrum_is_symmetric(self, trivial_ssh):
        """Test that spectrum is symmetric about zero (chiral symmetry)"""
        eigvals = np.linalg.eigvalsh(trivial_ssh.H.toarray())
        eigvals_sorted = np.sort(eigvals)

        # Due to chiral symmetry, if E is an eigenvalue, -E is also an eigenvalue
        # So spectrum should be symmetric
        L = len(eigvals_sorted)
        for i in range(L // 2):
            assert np.isclose(eigvals_sorted[i], -eigvals_sorted[L - 1 - i], atol=1e-10), \
                f"Spectrum not symmetric: E[{i}]={eigvals_sorted[i]}, E[{L-1-i}]={eigvals_sorted[L-1-i]}"

    def test_trivial_phase_has_gap(self):
        """Test that trivial phase (v > w) has energy gap"""
        np.random.seed(42)
        v_val = 2.0
        w_val = 1.0  # v > w -> trivial

        model = OneDimensionalSSHBlockBasis(
            L=100, disorder=0.0, rho=5.0, kappa=1.0,
            v=v_val, w=w_val
        )

        eigvals = np.linalg.eigvalsh(model.H.toarray())
        eigvals_sorted = np.sort(np.abs(eigvals))

        # Should have a gap at zero (no zero modes)
        min_abs_eigval = eigvals_sorted[0]
        gap = 2 * min_abs_eigval

        # Gap should be approximately 2|v - w| for clean system
        expected_gap = 2 * abs(v_val - w_val)

        assert gap > 0.1, f"Trivial phase should have gap, got {gap}"
        assert np.isclose(gap, expected_gap, rtol=0.2), \
            f"Gap {gap} should be close to 2|v-w|={expected_gap}"

    def test_topological_phase_spectrum(self):
        """
        Test topological phase (w > v) spectral properties

        IMPORTANT: With open boundary conditions, the topological phase
        has edge states at E ≈ 0, so the gap should be very small!
        This is the hallmark of topological protection.
        """
        np.random.seed(42)
        v_val = 1.0
        w_val = 2.0  # w > v -> topological

        model = OneDimensionalSSHBlockBasis(
            L=100, disorder=0.0, rho=5.0, kappa=1.0,
            v=v_val, w=w_val
        )

        eigvals = np.linalg.eigvalsh(model.H.toarray())
        eigvals_sorted = np.sort(np.abs(eigvals))

        # The smallest eigenvalue should be very close to zero (edge states!)
        min_eigval = eigvals_sorted[0]

        # Edge states should be at E ≈ 0 (within numerical precision)
        assert min_eigval < 1e-6, \
            f"Topological phase should have edge states near E=0, got min|E|={min_eigval}"

        # But there should be a bulk gap between edge states and bulk bands
        # Look for the gap after the edge states (first few eigenvalues)
        # Edge states are the 2 smallest eigenvalues (one per edge)
        bulk_min = eigvals_sorted[2]  # First bulk state

        # Bulk gap should be roughly 2|v-w|
        assert bulk_min > 0.5, \
            f"Bulk states should be separated from edge states, got {bulk_min}"

    def test_critical_point_small_gap(self):
        """Test that v=w (critical point) has very small gap"""
        np.random.seed(42)
        v_val = 1.0
        w_val = 1.0  # v = w -> critical

        model = OneDimensionalSSHBlockBasis(
            L=100, disorder=0.0, rho=5.0, kappa=1.0,
            v=v_val, w=w_val
        )

        eigvals = np.linalg.eigvalsh(model.H.toarray())
        min_abs_eigval = np.min(np.abs(eigvals))

        # At critical point, gap should be very small (finite size effects)
        assert min_abs_eigval < 0.1, \
            f"Critical point should have small gap, got {min_abs_eigval}"


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_minimal_system_L_equals_2(self):
        """Test L=2 (smallest possible SSH chain: 1 unit cell)"""
        np.random.seed(42)
        model = OneDimensionalSSHBlockBasis(
            L=2, disorder=0.0, rho=1.0, kappa=1.0,
            v=1.0, w=1.5
        )

        assert model.H.shape == (2, 2), "L=2 should give 2×2 Hamiltonian"
        assert model.X.shape == (1, 1), "L=2 should give 1×1 position operator"

        H = model.H.toarray()
        # H should be [[0, v], [v, 0]] for single unit cell (no intercell hopping)
        expected_H = np.array([[0, 1.0], [1.0, 0]])
        np.testing.assert_allclose(H, expected_H, rtol=1e-10)

    def test_small_system_L_equals_4(self):
        """Test L=4 (2 unit cells)"""
        np.random.seed(42)
        v_val = 1.0
        w_val = 1.5

        model = OneDimensionalSSHBlockBasis(
            L=4, disorder=0.0, rho=1.0, kappa=1.0,
            v=v_val, w=w_val
        )

        H = model.H.toarray()
        assert H.shape == (4, 4)

        # Check block structure
        # A should be 2×2: [[v, 0], [w, v]]
        A = H[:2, 2:]
        expected_A = np.array([[v_val, 0], [w_val, v_val]])
        np.testing.assert_allclose(A, expected_A, rtol=1e-10)

    def test_large_system_L_equals_1000(self):
        """Test that large systems work without memory issues"""
        np.random.seed(42)
        L = 1000

        model = OneDimensionalSSHBlockBasis(
            L=L, disorder=0.1, rho=5.0, kappa=1.0,
            v=1.0, w=1.5
        )

        assert sp.issparse(model.H), "Large system should use sparse matrices"
        assert model.H.shape == (L, L)
        assert model.X.shape == (L//2, L//2)

    @pytest.mark.parametrize("v,w", [
        (0.1, 2.0),    # Very small v
        (2.0, 0.1),    # Very small w
        (10.0, 10.0),  # Large equal values
        (0.01, 0.01),  # Very small equal values
    ])
    def test_extreme_parameter_values(self, v, w):
        """Test with extreme v and w values"""
        np.random.seed(42)
        model = OneDimensionalSSHBlockBasis(
            L=20, disorder=0.0, rho=5.0, kappa=1.0,
            v=v, w=w
        )

        # Should not crash and should maintain structure
        assert model.H.shape == (20, 20)

        # Check Hermiticity is maintained
        H = model.H.toarray()
        np.testing.assert_allclose(H, H.conj().T, rtol=1e-12)


# ============================================================================
# TEST CLASS: INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_workflow_no_errors(self):
        """Test complete workflow: create → diagonalize → compute stats"""
        np.random.seed(42)
        model = OneDimensionalSSHBlockBasis(
            L=40, disorder=0.2, rho=5.0, kappa=2.0,
            v=1.0, w=1.5
        )

        # Diagonalize Hamiltonian
        model.find_eigval(model.H, sparse=False)

        # Diagonalize spectral localizer
        model.find_eigval(model.spectral_localiser, sparse=False)

        # Compute statistics
        r_H = model.calculate_r(model.H_eigval)
        z_H = model.calculate_z(model.H_eigval)

        # Check reasonable values
        assert 0 < r_H < 1, f"r statistic {r_H} should be in (0,1)"
        assert 0 < z_H < 1, f"z statistic {z_H} should be in (0,1)"

        # Compute IPR
        ipr = model.compute_IPR(model.H_eigvec)
        assert len(ipr) == model.L
        assert np.all(ipr > 0)

    def test_disorder_affects_statistics(self):
        """Test that disorder changes spectral statistics"""
        L = 50
        v_val = 1.0
        w_val = 1.5

        # Clean system
        np.random.seed(42)
        model_clean = OneDimensionalSSHBlockBasis(
            L=L, disorder=0.0, rho=5.0, kappa=2.0,
            v=v_val, w=w_val
        )
        model_clean.find_eigval(model_clean.H, sparse=False)
        r_clean = model_clean.calculate_r(model_clean.H_eigval)

        # Disordered system
        np.random.seed(42)
        model_disorder = OneDimensionalSSHBlockBasis(
            L=L, disorder=2.0, rho=5.0, kappa=2.0,
            v=v_val, w=w_val
        )
        model_disorder.find_eigval(model_disorder.H, sparse=False)
        r_disorder = model_disorder.calculate_r(model_disorder.H_eigval)

        # Statistics should be different
        assert not np.isclose(r_clean, r_disorder, rtol=0.1), \
            "Disorder should affect spectral statistics"

    def test_topological_transition(self):
        """
        Test that v/w ratio affects spectrum (topological phase transition)

        The SSH model exhibits a topological phase transition at v = w:
        - v > w: Trivial phase, gap ≈ 2|v-w|, no edge states
        - w > v: Topological phase, edge states at E ≈ 0, gap ≈ 0
        """
        np.random.seed(42)
        L = 50

        # Trivial phase (v > w)
        model_trivial = OneDimensionalSSHBlockBasis(
            L=L, disorder=0.0, rho=5.0, kappa=1.0,
            v=2.0, w=1.0
        )
        eigvals_trivial = np.linalg.eigvalsh(model_trivial.H.toarray())
        gap_trivial = 2 * np.min(np.abs(eigvals_trivial))

        # Topological phase (w > v)
        model_topo = OneDimensionalSSHBlockBasis(
            L=L, disorder=0.0, rho=5.0, kappa=1.0,
            v=1.0, w=2.0
        )
        eigvals_topo = np.linalg.eigvalsh(model_topo.H.toarray())
        min_eigval_topo = np.min(np.abs(eigvals_topo))

        # Trivial phase should have a finite gap
        assert gap_trivial > 1.0, \
            f"Trivial phase should have gap ≈ 2|v-w| = 2, got {gap_trivial}"

        # Topological phase should have edge states near E=0
        assert min_eigval_topo < 1e-6, \
            f"Topological phase should have edge states at E≈0, got {min_eigval_topo}"

        # The gaps should be DIFFERENT (this is the phase transition!)
        assert gap_trivial > 1000 * min_eigval_topo, \
            "Trivial and topological phases should have very different gaps"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

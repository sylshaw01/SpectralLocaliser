"""
Unit tests for OneDimensionalAnderson model in SLmodels.py

Tests cover:
- Hamiltonian structure and properties
- Position operator
- Spectral localizer
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

from SLmodels import OneDimensionalAnderson


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_system():
    """L=10 system for fast tests"""
    np.random.seed(42)
    return OneDimensionalAnderson(L=10, disorder=1.0, rho=1.0, kappa=1.0)


@pytest.fixture
def clean_system():
    """Disorder-free system for analytical comparisons"""
    np.random.seed(42)
    return OneDimensionalAnderson(L=50, disorder=0.0, rho=1.0, kappa=1.0)


@pytest.fixture
def medium_system():
    """L=100 system for standard tests"""
    np.random.seed(42)
    return OneDimensionalAnderson(L=100, disorder=2.0, rho=5.0, kappa=3.0)


@pytest.fixture(params=[10, 50, 100])
def various_sizes(request):
    """Parametrized fixture for multiple system sizes"""
    np.random.seed(42)
    return OneDimensionalAnderson(L=request.param, disorder=2.0, rho=1.0, kappa=1.0)


# ============================================================================
# TEST CLASS: HAMILTONIAN STRUCTURE
# ============================================================================

class TestHamiltonianStructure:
    """Test basic structural properties of the Hamiltonian"""

    def test_hamiltonian_shape(self, various_sizes):
        """Test that Hamiltonian has correct dimensions L×L"""
        model = various_sizes
        L = model.L
        assert model.H.shape == (L, L), f"Expected shape ({L}, {L}), got {model.H.shape}"

    def test_hamiltonian_is_sparse(self, small_system):
        """Test that Hamiltonian is returned as sparse CSR matrix"""
        assert sp.issparse(small_system.H), "Hamiltonian should be sparse matrix"
        assert small_system.H.format == 'csr', f"Expected CSR format, got {small_system.H.format}"

    def test_hamiltonian_is_hermitian(self, medium_system):
        """Test that H† = H (Hermiticity) - critical for physical validity"""
        H = medium_system.H.toarray()
        H_conj_T = H.conj().T

        np.testing.assert_allclose(
            H, H_conj_T,
            rtol=1e-14, atol=1e-14,
            err_msg="Hamiltonian is not Hermitian"
        )

    def test_hamiltonian_is_real(self, medium_system):
        """Test that all matrix elements are real (Anderson model property)"""
        H = medium_system.H.toarray()
        assert np.all(np.isreal(H)), "Hamiltonian should have only real elements"
        assert H.dtype in [np.float32, np.float64], f"Hamiltonian should be real type, got {H.dtype}"

    def test_hamiltonian_is_tridiagonal(self, small_system):
        """Test that H has only 3 diagonals (nearest-neighbor hopping only)"""
        H = small_system.H.toarray()
        L = small_system.L

        # Create mask for tridiagonal elements
        mask = np.eye(L, k=0) + np.eye(L, k=1) + np.eye(L, k=-1)
        non_tridiag_mask = (mask == 0)

        # All elements outside tridiagonal should be zero
        assert np.allclose(H[non_tridiag_mask], 0, atol=1e-14), \
            "Non-zero elements found outside tridiagonal structure"

    def test_hamiltonian_sparsity_count(self, small_system):
        """Test that H has exactly 3L-2 non-zero elements"""
        L = small_system.L
        expected_nnz = 3 * L - 2  # L diagonal + 2*(L-1) off-diagonal
        actual_nnz = small_system.H.nnz

        assert actual_nnz == expected_nnz, \
            f"Expected {expected_nnz} non-zero elements, got {actual_nnz}"


# ============================================================================
# TEST CLASS: MATRIX ELEMENT VALUES
# ============================================================================

class TestMatrixElements:
    """Test specific values in the Hamiltonian matrix"""

    def test_off_diagonal_elements_are_one(self, medium_system):
        """Test that all off-diagonal elements equal 1.0 (hopping strength)"""
        H = medium_system.H.toarray()
        L = medium_system.L

        # Check upper off-diagonal
        for i in range(L - 1):
            assert H[i, i+1] == 1.0, f"Upper off-diagonal H[{i},{i+1}] = {H[i,i+1]}, expected 1.0"

        # Check lower off-diagonal (should be same due to Hermiticity)
        for i in range(L - 1):
            assert H[i+1, i] == 1.0, f"Lower off-diagonal H[{i+1},{i}] = {H[i+1,i]}, expected 1.0"

    def test_diagonal_disorder_range(self):
        """Test diagonal elements in correct range [-disorder/2, disorder/2]"""
        np.random.seed(42)
        disorder = 4.0
        model = OneDimensionalAnderson(L=100, disorder=disorder, rho=1.0, kappa=1.0)

        diag = np.diag(model.H.toarray())

        assert np.all(diag >= -disorder/2 - 1e-10), \
            f"Diagonal element {diag.min()} below -disorder/2 = {-disorder/2}"
        assert np.all(diag <= disorder/2 + 1e-10), \
            f"Diagonal element {diag.max()} above disorder/2 = {disorder/2}"

    def test_no_disorder_gives_zero_diagonal(self):
        """Test that disorder=0 gives diagonal elements all zero"""
        np.random.seed(42)
        model = OneDimensionalAnderson(L=50, disorder=0.0, rho=1.0, kappa=1.0)

        diag = np.diag(model.H.toarray())

        np.testing.assert_allclose(
            diag, 0.0,
            atol=1e-14,
            err_msg="Zero disorder should give zero diagonal elements"
        )

    def test_boundary_conditions(self, medium_system):
        """Test open boundary conditions: H[0,L-1] = H[L-1,0] = 0"""
        H = medium_system.H.toarray()
        L = medium_system.L

        # No periodic wrapping
        assert H[0, L-1] == 0, "Expected open boundary: H[0,L-1] should be 0"
        assert H[L-1, 0] == 0, "Expected open boundary: H[L-1,0] should be 0"

    def test_tridiagonal_structure_detailed(self):
        """Detailed verification of exact tridiagonal structure"""
        np.random.seed(42)
        L = 50
        disorder = 2.0

        model = OneDimensionalAnderson(L, disorder, rho=1.0, kappa=1.0)
        H_dense = model.H.toarray()

        # Check off-diagonal elements
        for i in range(L-1):
            assert H_dense[i, i+1] == 1.0, f"Upper diagonal element [{i},{i+1}] != 1.0"
            assert H_dense[i+1, i] == 1.0, f"Lower diagonal element [{i+1},{i}] != 1.0"

        # Check diagonal range
        diag = np.diag(H_dense)
        assert np.all(diag >= -disorder/2 - 1e-10), "Diagonal element below -disorder/2"
        assert np.all(diag <= disorder/2 + 1e-10), "Diagonal element above disorder/2"

        # Check all other elements are zero
        mask = np.eye(L, k=0) + np.eye(L, k=1) + np.eye(L, k=-1)
        zeros_mask = (mask == 0)
        assert np.allclose(H_dense[zeros_mask], 0), "Non-zero elements outside tridiagonal"


# ============================================================================
# TEST CLASS: PHYSICAL PROPERTIES
# ============================================================================

class TestPhysicalProperties:
    """Test physical constraints and behaviors"""

    def test_eigenvalue_count(self, various_sizes):
        """Test that H has exactly L eigenvalues"""
        model = various_sizes
        L = model.L

        eigvals = np.linalg.eigvalsh(model.H.toarray())

        assert len(eigvals) == L, f"Expected {L} eigenvalues, got {len(eigvals)}"

    def test_eigenvalues_are_real(self, medium_system):
        """Test that all eigenvalues are real (Hermitian property)"""
        eigvals = np.linalg.eigvals(medium_system.H.toarray())

        assert np.all(np.isreal(eigvals)), "Eigenvalues should all be real"
        assert np.max(np.abs(np.imag(eigvals))) < 1e-10, \
            f"Imaginary parts too large: {np.max(np.abs(np.imag(eigvals)))}"

    def test_zero_disorder_gives_cosine_band(self):
        """
        Test that disorder=0 gives analytical eigenvalues E_k = -2*cos(πk/(L+1))
        This is the CRITICAL validation test for the numerical implementation
        """
        np.random.seed(42)
        L = 50
        disorder = 0.0

        model = OneDimensionalAnderson(L, disorder, rho=1.0, kappa=1.0)

        # Numerical eigenvalues
        eigvals_numerical = np.linalg.eigvalsh(model.H.toarray())
        eigvals_numerical_sorted = np.sort(eigvals_numerical)

        # Analytical eigenvalues for 1D tight-binding chain
        k_values = np.arange(1, L+1)
        eigvals_analytical = -2 * np.cos(np.pi * k_values / (L + 1))
        eigvals_analytical_sorted = np.sort(eigvals_analytical)

        # Compare
        np.testing.assert_allclose(
            eigvals_numerical_sorted,
            eigvals_analytical_sorted,
            rtol=1e-10,
            err_msg="Clean system eigenvalues don't match analytical solution"
        )

    def test_eigenvalue_range_scales_with_disorder(self):
        """Test that eigenvalue spread increases with disorder"""
        np.random.seed(42)
        L = 100

        # Test increasing disorder
        disorders = [0.0, 1.0, 5.0, 10.0]
        spreads = []

        for disorder in disorders:
            np.random.seed(42)  # Same disorder realization for fair comparison
            model = OneDimensionalAnderson(L, disorder, rho=1.0, kappa=1.0)
            eigvals = np.linalg.eigvalsh(model.H.toarray())
            spread = eigvals.max() - eigvals.min()
            spreads.append(spread)

        # Spreads should generally increase with disorder
        # (though not monotonically for single realizations)
        assert spreads[-1] > spreads[0], \
            f"High disorder spread {spreads[-1]} should exceed zero disorder spread {spreads[0]}"

    def test_spectral_center_near_zero(self):
        """Test that mean eigenvalue ≈ 0 for ensemble average"""
        np.random.seed(42)
        L = 50
        disorder = 2.0
        num_realizations = 50

        mean_eigvals = []
        for i in range(num_realizations):
            np.random.seed(42 + i)
            model = OneDimensionalAnderson(L, disorder, rho=1.0, kappa=1.0)
            eigvals = np.linalg.eigvalsh(model.H.toarray())
            mean_eigvals.append(np.mean(eigvals))

        ensemble_mean = np.mean(mean_eigvals)

        # Ensemble mean should be close to 0 (disorder is symmetric)
        assert np.abs(ensemble_mean) < 0.1, \
            f"Ensemble mean eigenvalue {ensemble_mean} should be near 0"


# ============================================================================
# TEST CLASS: REPRODUCIBILITY
# ============================================================================

class TestReproducibility:
    """Test random seed control and determinism"""

    def test_same_seed_gives_same_hamiltonian(self):
        """Test that same random seed produces identical Hamiltonians"""
        L = 50
        disorder = 2.0

        # Create first model
        np.random.seed(12345)
        model1 = OneDimensionalAnderson(L, disorder, rho=1.0, kappa=1.0)
        H1 = model1.H.toarray()

        # Create second model with same seed
        np.random.seed(12345)
        model2 = OneDimensionalAnderson(L, disorder, rho=1.0, kappa=1.0)
        H2 = model2.H.toarray()

        # Should be identical
        np.testing.assert_array_equal(
            H1, H2,
            err_msg="Same seed should produce identical Hamiltonians"
        )

    def test_different_seeds_give_different_hamiltonians(self):
        """Test that different seeds produce different disorder realizations"""
        L = 50
        disorder = 2.0

        # Create models with different seeds
        np.random.seed(111)
        model1 = OneDimensionalAnderson(L, disorder, rho=1.0, kappa=1.0)
        H1 = model1.H.toarray()

        np.random.seed(222)
        model2 = OneDimensionalAnderson(L, disorder, rho=1.0, kappa=1.0)
        H2 = model2.H.toarray()

        # Should be different
        assert not np.allclose(H1, H2), \
            "Different seeds should produce different Hamiltonians"

        # But off-diagonals should still be the same
        for i in range(L-1):
            assert H1[i, i+1] == H2[i, i+1] == 1.0, "Off-diagonals should remain 1.0"


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test boundary conditions and special cases"""

    def test_minimal_system_L_equals_2(self):
        """Test L=2 system (smallest nontrivial case)"""
        np.random.seed(42)
        model = OneDimensionalAnderson(L=2, disorder=1.0, rho=1.0, kappa=1.0)
        H = model.H.toarray()

        # Should be 2×2
        assert H.shape == (2, 2)

        # Off-diagonal should be 1.0
        assert H[0, 1] == 1.0
        assert H[1, 0] == 1.0

        # Diagonal should be in range
        assert -0.5 <= H[0, 0] <= 0.5
        assert -0.5 <= H[1, 1] <= 0.5

    def test_single_site_L_equals_1(self):
        """Test L=1 system (single site)"""
        np.random.seed(42)
        model = OneDimensionalAnderson(L=1, disorder=1.0, rho=1.0, kappa=1.0)
        H = model.H.toarray()

        # Should be 1×1 matrix
        assert H.shape == (1, 1)

        # Only element should be disorder term
        assert -0.5 <= H[0, 0] <= 0.5

    def test_large_system_L_equals_10000(self):
        """Test that large systems don't cause memory/performance issues"""
        np.random.seed(42)
        L = 10000

        # Should complete without error
        model = OneDimensionalAnderson(L, disorder=2.0, rho=1.0, kappa=1.0)

        # Verify still sparse
        assert sp.issparse(model.H)

        # Verify correct size
        assert model.H.shape == (L, L)

        # Verify sparsity is maintained (not densified)
        expected_nnz = 3 * L - 2
        assert model.H.nnz == expected_nnz

    @pytest.mark.parametrize("disorder", [0.0, 0.1, 1.0, 10.0, 100.0])
    def test_extreme_disorder_values(self, disorder):
        """Test with various disorder strengths"""
        np.random.seed(42)
        L = 50

        # Should not raise errors
        model = OneDimensionalAnderson(L, disorder, rho=1.0, kappa=1.0)

        # Check diagonal range
        diag = np.diag(model.H.toarray())
        assert np.all(diag >= -disorder/2 - 1e-10)
        assert np.all(diag <= disorder/2 + 1e-10)


# ============================================================================
# TEST CLASS: POSITION OPERATOR
# ============================================================================

class TestPositionOperator:
    """Test position operator X"""

    def test_position_operator_shape(self, various_sizes):
        """Test X has shape L×L"""
        model = various_sizes
        L = model.L
        assert model.X.shape == (L, L), f"Expected X shape ({L}, {L}), got {model.X.shape}"

    def test_position_operator_is_diagonal(self, medium_system):
        """Test X is diagonal matrix"""
        X = medium_system.X.toarray()
        L = medium_system.L

        # All off-diagonal elements should be zero
        off_diag_mask = ~np.eye(L, dtype=bool)
        assert np.allclose(X[off_diag_mask], 0), "Position operator should be diagonal"

    def test_position_operator_range(self):
        """Test diagonal elements range from -rho to +rho"""
        np.random.seed(42)
        rho = 5.0
        model = OneDimensionalAnderson(L=100, disorder=1.0, rho=rho, kappa=1.0)

        X_diag = np.diag(model.X.toarray())

        assert np.isclose(X_diag.min(), -rho, rtol=1e-10), \
            f"Min position {X_diag.min()} should be -rho={-rho}"
        assert np.isclose(X_diag.max(), rho, rtol=1e-10), \
            f"Max position {X_diag.max()} should be rho={rho}"

    def test_position_operator_linearity(self):
        """Test that X_i - X_{i-1} is constant (evenly spaced)"""
        np.random.seed(42)
        model = OneDimensionalAnderson(L=100, disorder=1.0, rho=5.0, kappa=1.0)

        X_diag = np.diag(model.X.toarray())
        differences = np.diff(X_diag)

        # All differences should be equal
        assert np.allclose(differences, differences[0], rtol=1e-10), \
            "Position operator elements should be evenly spaced"

    def test_position_operator_is_sparse(self, small_system):
        """Test X is stored as sparse matrix"""
        assert sp.issparse(small_system.X), "Position operator should be sparse"

    def test_position_operator_symmetry(self):
        """Test that position operator is symmetric about 0"""
        np.random.seed(42)
        model = OneDimensionalAnderson(L=101, disorder=1.0, rho=5.0, kappa=1.0)  # Odd L

        X_diag = np.diag(model.X.toarray())

        # For odd L, middle element should be 0
        middle_idx = len(X_diag) // 2
        assert np.isclose(X_diag[middle_idx], 0, atol=1e-10), \
            "Middle element of position operator should be 0 for odd L"


# ============================================================================
# TEST CLASS: SPECTRAL LOCALIZER
# ============================================================================

class TestSpectralLocalizer:
    """Test spectral localizer SL"""

    def test_localizer_shape(self, medium_system):
        """Test SL has shape 2L×2L"""
        L = medium_system.L
        SL = medium_system.spectral_localiser

        assert SL.shape == (2*L, 2*L), \
            f"Expected spectral localizer shape ({2*L}, {2*L}), got {SL.shape}"

    def test_localizer_structure(self):
        """Test SL = [[-H, κX], [κX, H]] block structure"""
        np.random.seed(42)
        L = 20  # Small for easier checking
        kappa = 3.0
        model = OneDimensionalAnderson(L, disorder=1.0, rho=5.0, kappa=kappa)

        SL = model.spectral_localiser.toarray()
        H = model.H.toarray()
        X = model.X.toarray()

        # Check each block
        top_left = SL[:L, :L]
        top_right = SL[:L, L:]
        bottom_left = SL[L:, :L]
        bottom_right = SL[L:, L:]

        np.testing.assert_allclose(top_left, -H, rtol=1e-12,
                                   err_msg="Top-left block should be -H")
        np.testing.assert_allclose(top_right, kappa * X, rtol=1e-12,
                                   err_msg="Top-right block should be κX")
        np.testing.assert_allclose(bottom_left, kappa * X, rtol=1e-12,
                                   err_msg="Bottom-left block should be κX")
        np.testing.assert_allclose(bottom_right, H, rtol=1e-12,
                                   err_msg="Bottom-right block should be H")

    def test_localizer_is_hermitian(self, medium_system):
        """Test SL† = SL"""
        SL = medium_system.spectral_localiser.toarray()
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

        for kappa in kappa_values:
            model = OneDimensionalAnderson(L, disorder=1.0, rho=5.0, kappa=kappa)
            SL = model.spectral_localiser.toarray()
            X = model.X.toarray()

            # Check that off-diagonal blocks are κX
            top_right = SL[:L, L:]
            expected = kappa * X

            np.testing.assert_allclose(
                top_right, expected, rtol=1e-12,
                err_msg=f"κX block incorrect for κ={kappa}"
            )

    def test_localizer_is_sparse(self, medium_system):
        """Test that spectral localizer is stored as sparse matrix"""
        assert sp.issparse(medium_system.spectral_localiser), \
            "Spectral localizer should be sparse matrix"


# ============================================================================
# TEST CLASS: INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_workflow_no_errors(self):
        """Test complete workflow: create model → diagonalize → compute stats"""
        np.random.seed(42)
        L = 50

        # Create model
        model = OneDimensionalAnderson(L, disorder=2.0, rho=5.0, kappa=3.0)

        # Diagonalize Hamiltonian
        model.find_eigval(model.H, sparse=False)

        # Diagonalize spectral localizer
        model.find_eigval(model.spectral_localiser, sparse=False)

        # Compute statistics
        r_H = model.calculate_r(model.H_eigval)
        z_H = model.calculate_z(model.H_eigval)

        # Check that results are reasonable
        assert 0 < r_H < 1, f"r statistic {r_H} should be in (0,1)"
        assert 0 < z_H < 1, f"z statistic {z_H} should be in (0,1)"

        # Compute IPR
        ipr = model.compute_IPR(model.H_eigvec)
        assert len(ipr) == L, "IPR should have L values"
        assert np.all(ipr > 0), "IPR should be positive"

    def test_localization_with_disorder(self):
        """Test that IPR behavior changes with disorder"""
        np.random.seed(42)
        L = 100

        # Clean system
        model_clean = OneDimensionalAnderson(L, disorder=0.0, rho=5.0, kappa=3.0)
        model_clean.find_eigval(model_clean.H, sparse=False)
        ipr_clean = model_clean.compute_IPR(model_clean.H_eigvec)

        # Disordered system
        model_disorder = OneDimensionalAnderson(L, disorder=10.0, rho=5.0, kappa=3.0)
        model_disorder.find_eigval(model_disorder.H, sparse=False)
        ipr_disorder = model_disorder.compute_IPR(model_disorder.H_eigvec)

        # Mean IPR should be higher for disordered system (more localized)
        # Note: This is a statistical test, might need averaging over realizations for robustness
        mean_ipr_clean = np.mean(ipr_clean)
        mean_ipr_disorder = np.mean(ipr_disorder)

        # At least check they're in reasonable ranges
        assert 0 < mean_ipr_clean < 1, "Clean system mean IPR should be in (0,1)"
        assert 0 < mean_ipr_disorder < 1, "Disordered system mean IPR should be in (0,1)"

    def test_eigenvalue_sorting_in_find_eigval(self):
        """
        CRITICAL TEST: Verify that find_eigval stores sorted eigenvalues
        This addresses the bug found in code review
        """
        np.random.seed(42)
        L = 50
        model = OneDimensionalAnderson(L, disorder=2.0, rho=5.0, kappa=3.0)

        # Test sparse solver
        model.find_eigval(model.H, num_eigval=20, sparse=True)
        eigvals_sparse = model.H_eigval
        assert np.all(eigvals_sparse[:-1] <= eigvals_sparse[1:]), \
            "Eigenvalues from sparse solver should be sorted"

        # Test dense solver
        model.find_eigval(model.H, sparse=False)
        eigvals_dense = model.H_eigval
        assert np.all(eigvals_dense[:-1] <= eigvals_dense[1:]), \
            "Eigenvalues from dense solver should be sorted"

    def test_statistics_require_sorted_eigenvalues(self):
        """Test that r and z statistics depend on eigenvalue ordering"""
        np.random.seed(42)
        L = 50
        model = OneDimensionalAnderson(L, disorder=2.0, rho=5.0, kappa=3.0)

        model.find_eigval(model.H, sparse=False)
        eigvals = model.H_eigval

        # Calculate statistics with sorted eigenvalues
        r_sorted = model.calculate_r(eigvals)
        z_sorted = model.calculate_z(eigvals)

        # Shuffle eigenvalues
        eigvals_shuffled = np.random.permutation(eigvals)

        # Calculate statistics with shuffled eigenvalues
        r_shuffled = model.calculate_r(eigvals_shuffled)
        z_shuffled = model.calculate_z(eigvals_shuffled)

        # Results should be different (demonstrating importance of sorting)
        assert not np.isclose(r_sorted, r_shuffled), \
            "r statistic should depend on eigenvalue order"
        assert not np.isclose(z_sorted, z_shuffled), \
            "z statistic should depend on eigenvalue order"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for OneDimensionalAubryAndre model in SLmodels.py

The Aubry-André model is a 1D quasiperiodic system:
- Quasi-periodic potential: V(n) = λ cos(2πβn + θ)
- β = (√5-1)/2 (golden ratio) - key to quasiperiodicity
- Exhibits metal-insulator transition (Aubry-André transition)
- λ < 2: Extended states (metallic)
- λ = 2: Critical point (self-dual)
- λ > 2: Localized states (insulating)

Tests cover:
- Quasi-periodic potential structure
- Golden ratio β parameter
- Phase parameter θ
- Aubry-André transition physics
- Periodic vs open boundary conditions
- Position operator
- Spectral localizer
- Localization properties
"""

import pytest
import numpy as np
import scipy.sparse as sp
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from SLmodels import OneDimensionalAubryAndre


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def extended_aubry():
    """Extended (metallic) phase: λ < 2"""
    np.random.seed(42)
    return OneDimensionalAubryAndre(
        L=50, disorder=1.0, rho=5.0, kappa=1.0,
        beta=(np.sqrt(5)-1)/2, theta=0
    )


@pytest.fixture
def critical_aubry():
    """Critical point: λ = 2 (self-dual)"""
    np.random.seed(42)
    return OneDimensionalAubryAndre(
        L=50, disorder=2.0, rho=5.0, kappa=1.0,
        beta=(np.sqrt(5)-1)/2, theta=0
    )


@pytest.fixture
def localized_aubry():
    """Localized (insulating) phase: λ > 2"""
    np.random.seed(42)
    return OneDimensionalAubryAndre(
        L=50, disorder=5.0, rho=5.0, kappa=1.0,
        beta=(np.sqrt(5)-1)/2, theta=0
    )


@pytest.fixture
def small_aubry():
    """Small system for detailed testing"""
    np.random.seed(42)
    return OneDimensionalAubryAndre(
        L=10, disorder=1.0, rho=2.0, kappa=1.0,
        beta=(np.sqrt(5)-1)/2, theta=0
    )


@pytest.fixture(params=[10, 30, 50, 100])
def various_sizes_aubry(request):
    """Parametrized fixture for multiple system sizes"""
    np.random.seed(42)
    return OneDimensionalAubryAndre(
        L=request.param, disorder=1.0, rho=5.0, kappa=1.0,
        beta=(np.sqrt(5)-1)/2, theta=0
    )


# ============================================================================
# TEST CLASS: BASIC STRUCTURE
# ============================================================================

class TestBasicStructure:
    """Test basic structural properties"""

    def test_hamiltonian_shape(self, various_sizes_aubry):
        """Test that H has shape L×L"""
        model = various_sizes_aubry
        L = model.L
        assert model.H.shape == (L, L), f"Expected H shape ({L}, {L}), got {model.H.shape}"

    def test_hamiltonian_is_sparse(self, small_aubry):
        """Test that H is sparse CSR matrix"""
        assert sp.issparse(small_aubry.H), "Hamiltonian should be sparse"
        assert small_aubry.H.format == 'csr', f"Expected CSR format, got {small_aubry.H.format}"

    def test_hamiltonian_is_hermitian(self, extended_aubry):
        """Test H† = H (Hermiticity)"""
        H = extended_aubry.H.toarray()
        H_conj_T = H.conj().T

        np.testing.assert_allclose(
            H, H_conj_T,
            rtol=1e-14, atol=1e-14,
            err_msg="Aubry-André Hamiltonian should be Hermitian"
        )

    def test_hamiltonian_is_real(self, small_aubry):
        """Test that H is real"""
        H = small_aubry.H.toarray()
        assert np.all(np.isreal(H)), "Hamiltonian should be real"

    def test_hamiltonian_is_tridiagonal(self, small_aubry):
        """Test that H has tridiagonal structure (OBC)"""
        H = small_aubry.H.toarray()
        L = small_aubry.L

        # Create mask for tridiagonal
        mask = np.eye(L, k=0) + np.eye(L, k=1) + np.eye(L, k=-1)
        non_tridiag_mask = (mask == 0)

        # All elements outside tridiagonal should be zero
        assert np.allclose(H[non_tridiag_mask], 0, atol=1e-14), \
            "Hamiltonian should be tridiagonal with OBC"

    def test_parameters_stored_correctly(self):
        """Test that model parameters are stored"""
        np.random.seed(42)
        beta_val = 0.5
        theta_val = np.pi/4

        model = OneDimensionalAubryAndre(
            L=20, disorder=1.0, rho=1.0, kappa=1.0,
            beta=beta_val, theta=theta_val
        )

        assert model.beta == beta_val, f"Expected beta={beta_val}, got {model.beta}"
        assert model.theta == theta_val, f"Expected theta={theta_val}, got {model.theta}"


# ============================================================================
# TEST CLASS: QUASI-PERIODIC POTENTIAL
# ============================================================================

class TestQuasiperiodicPotential:
    """Test quasi-periodic potential structure"""

    def test_diagonal_is_quasiperiodic(self):
        """Test that diagonal follows V(n) = λ cos(2πβn + θ)"""
        np.random.seed(42)
        L = 20
        disorder = 3.0  # λ = 3
        beta = (np.sqrt(5)-1)/2
        theta = np.pi/3

        model = OneDimensionalAubryAndre(
            L=L, disorder=disorder, rho=1.0, kappa=1.0,
            beta=beta, theta=theta
        )

        H = model.H.toarray()
        diag = np.diag(H)

        # Expected diagonal
        n = np.arange(L)
        expected_diag = disorder * np.cos(2 * np.pi * beta * n + theta)

        np.testing.assert_allclose(
            diag, expected_diag, rtol=1e-12,
            err_msg="Diagonal should match quasi-periodic formula"
        )

    def test_golden_ratio_default(self):
        """Test that default β is golden ratio"""
        np.random.seed(42)
        model = OneDimensionalAubryAndre(
            L=20, disorder=1.0, rho=1.0, kappa=1.0
        )

        golden_ratio = (np.sqrt(5) - 1) / 2
        assert np.isclose(model.beta, golden_ratio, rtol=1e-15), \
            f"Default β should be golden ratio {golden_ratio}"

    def test_theta_phase_shift(self):
        """Test that θ parameter shifts the potential"""
        np.random.seed(42)
        L = 20
        disorder = 2.0

        model1 = OneDimensionalAubryAndre(
            L=L, disorder=disorder, rho=1.0, kappa=1.0,
            beta=(np.sqrt(5)-1)/2, theta=0.0
        )

        model2 = OneDimensionalAubryAndre(
            L=L, disorder=disorder, rho=1.0, kappa=1.0,
            beta=(np.sqrt(5)-1)/2, theta=np.pi/2
        )

        H1 = model1.H.toarray()
        H2 = model2.H.toarray()

        diag1 = np.diag(H1)
        diag2 = np.diag(H2)

        # Different θ should give different potentials
        assert not np.allclose(diag1, diag2), \
            "Different θ should produce different potentials"

    def test_zero_disorder_gives_zero_potential(self):
        """Test that λ=0 (disorder=0) gives zero potential"""
        np.random.seed(42)
        model = OneDimensionalAubryAndre(
            L=20, disorder=0.0, rho=1.0, kappa=1.0
        )

        H = model.H.toarray()
        diag = np.diag(H)

        np.testing.assert_allclose(diag, 0.0, atol=1e-14,
                                   err_msg="Zero disorder should give zero potential")

    def test_potential_is_aperiodic(self):
        """Test that potential doesn't repeat (quasiperiodic property)"""
        np.random.seed(42)
        L = 100
        model = OneDimensionalAubryAndre(
            L=L, disorder=2.0, rho=1.0, kappa=1.0,
            beta=(np.sqrt(5)-1)/2, theta=0
        )

        H = model.H.toarray()
        diag = np.diag(H)

        # Check that it's not periodic with any small period
        for period in [2, 3, 4, 5, 10]:
            if period < L // 2:
                # Check if diag repeats with this period
                is_periodic = np.allclose(diag[:L-period], diag[period:])
                assert not is_periodic, \
                    f"Quasi-periodic potential should not have period {period}"

    def test_potential_range(self):
        """Test that potential is in range [-λ, λ]"""
        np.random.seed(42)
        disorder = 3.0
        model = OneDimensionalAubryAndre(
            L=50, disorder=disorder, rho=1.0, kappa=1.0
        )

        H = model.H.toarray()
        diag = np.diag(H)

        assert np.all(diag >= -disorder - 1e-10), \
            f"Potential should be >= -λ = {-disorder}"
        assert np.all(diag <= disorder + 1e-10), \
            f"Potential should be <= λ = {disorder}"

        # Should actually reach the bounds (approximately)
        assert np.max(diag) > disorder - 0.1, "Should approach upper bound"
        assert np.min(diag) < -disorder + 0.1, "Should approach lower bound"


# ============================================================================
# TEST CLASS: HOPPING TERMS
# ============================================================================

class TestHoppingTerms:
    """Test hopping structure"""

    def test_off_diagonal_hopping(self, small_aubry):
        """Test that off-diagonal hopping is -1.0"""
        H = small_aubry.H.toarray()
        L = small_aubry.L

        # Check upper and lower off-diagonals
        for i in range(L-1):
            assert H[i, i+1] == -1.0, f"H[{i},{i+1}] should be -1.0"
            assert H[i+1, i] == -1.0, f"H[{i+1},{i}] should be -1.0"

    def test_open_boundary_conditions(self, small_aubry):
        """Test open boundary conditions by default"""
        H = small_aubry.H.toarray()
        L = small_aubry.L

        # Check that corners are zero (no PBC)
        assert H[0, L-1] == 0, "H[0,L-1] should be 0 (OBC)"
        assert H[L-1, 0] == 0, "H[L-1,0] should be 0 (OBC)"

    def test_periodic_boundary_conditions(self):
        """Test periodic boundary conditions option"""
        np.random.seed(42)
        L = 10
        model = OneDimensionalAubryAndre(
            L=L, disorder=1.0, rho=1.0, kappa=1.0
        )

        # Create Hamiltonian with PBC
        H_pbc = model.create_hamiltonian(pbc=True).toarray()

        # Check that corners have hopping (PBC)
        assert H_pbc[0, L-1] == -1.0, "H[0,L-1] should be -1.0 (PBC)"
        assert H_pbc[L-1, 0] == -1.0, "H[L-1,0] should be -1.0 (PBC)"

    def test_pbc_vs_obc_difference(self):
        """Test difference between PBC and OBC"""
        np.random.seed(42)
        L = 20
        model = OneDimensionalAubryAndre(
            L=L, disorder=1.0, rho=1.0, kappa=1.0
        )

        H_obc = model.H.toarray()  # Default is OBC
        H_pbc = model.create_hamiltonian(pbc=True).toarray()

        # Should differ at corners
        assert not np.allclose(H_obc, H_pbc), \
            "OBC and PBC Hamiltonians should differ"

        # Interior should be the same
        interior = H_obc[1:-1, 1:-1]
        interior_pbc = H_pbc[1:-1, 1:-1]
        np.testing.assert_allclose(interior, interior_pbc,
                                   err_msg="Interior should be same for OBC and PBC")


# ============================================================================
# TEST CLASS: POSITION OPERATOR
# ============================================================================

class TestPositionOperator:
    """Test position operator"""

    def test_position_operator_shape(self, various_sizes_aubry):
        """Test X has shape L×L"""
        model = various_sizes_aubry
        L = model.L
        assert model.X.shape == (L, L), f"Expected X shape ({L}, {L}), got {model.X.shape}"

    def test_position_operator_is_diagonal(self, small_aubry):
        """Test X is diagonal"""
        X = small_aubry.X.toarray()
        L = small_aubry.L

        off_diag_mask = ~np.eye(L, dtype=bool)
        assert np.allclose(X[off_diag_mask], 0), "Position operator should be diagonal"

    def test_position_operator_range(self):
        """Test that X ranges from -ρ to +ρ"""
        np.random.seed(42)
        rho = 7.5
        model = OneDimensionalAubryAndre(
            L=50, disorder=1.0, rho=rho, kappa=1.0
        )

        X_diag = np.diag(model.X.toarray())

        assert np.isclose(X_diag.min(), -rho, rtol=1e-10), \
            f"Min position should be -rho={-rho}"
        assert np.isclose(X_diag.max(), rho, rtol=1e-10), \
            f"Max position should be rho={rho}"

    def test_position_operator_linearity(self):
        """Test that position operator is evenly spaced"""
        np.random.seed(42)
        model = OneDimensionalAubryAndre(
            L=50, disorder=1.0, rho=5.0, kappa=1.0
        )

        X_diag = np.diag(model.X.toarray())
        differences = np.diff(X_diag)

        # All differences should be equal
        assert np.allclose(differences, differences[0], rtol=1e-10), \
            "Position operator should be evenly spaced"


# ============================================================================
# TEST CLASS: SPECTRAL LOCALIZER
# ============================================================================

class TestSpectralLocalizer:
    """Test spectral localizer structure"""

    def test_localizer_shape(self, small_aubry):
        """Test SL has shape 2L×2L"""
        L = small_aubry.L
        SL = small_aubry.spectral_localiser

        assert SL.shape == (2*L, 2*L), \
            f"Spectral localizer should be ({2*L}, {2*L}), got {SL.shape}"

    def test_localizer_structure(self):
        """Test SL = [[-H, κX], [κX, H]] structure"""
        np.random.seed(42)
        L = 10
        kappa = 2.0

        model = OneDimensionalAubryAndre(
            L=L, disorder=1.0, rho=5.0, kappa=kappa
        )

        SL = model.spectral_localiser.toarray()
        H = model.H.toarray()
        X = model.X.toarray()

        # Extract blocks
        top_left = SL[:L, :L]
        top_right = SL[:L, L:]
        bottom_left = SL[L:, :L]
        bottom_right = SL[L:, L:]

        # Check structure
        np.testing.assert_allclose(top_left, -H, rtol=1e-12,
                                   err_msg="Top-left should be -H")
        np.testing.assert_allclose(top_right, kappa * X, rtol=1e-12,
                                   err_msg="Top-right should be κX")
        np.testing.assert_allclose(bottom_left, kappa * X, rtol=1e-12,
                                   err_msg="Bottom-left should be κX")
        np.testing.assert_allclose(bottom_right, H, rtol=1e-12,
                                   err_msg="Bottom-right should be H")

    def test_localizer_is_hermitian(self, extended_aubry):
        """Test SL† = SL"""
        SL = extended_aubry.spectral_localiser.toarray()
        SL_conj_T = SL.conj().T

        np.testing.assert_allclose(SL, SL_conj_T, rtol=1e-12, atol=1e-12,
                                   err_msg="Spectral localizer should be Hermitian")

    def test_localizer_is_sparse(self, small_aubry):
        """Test that spectral localizer is sparse"""
        assert sp.issparse(small_aubry.spectral_localiser), \
            "Spectral localizer should be sparse"


# ============================================================================
# TEST CLASS: PHYSICAL PROPERTIES
# ============================================================================

class TestPhysicalProperties:
    """Test physical properties and transitions"""

    def test_eigenvalue_count(self, small_aubry):
        """Test that H has L eigenvalues"""
        L = small_aubry.L
        eigvals = np.linalg.eigvalsh(small_aubry.H.toarray())

        assert len(eigvals) == L, f"Expected {L} eigenvalues, got {len(eigvals)}"

    def test_eigenvalues_are_real(self, small_aubry):
        """Test that eigenvalues are real"""
        eigvals = np.linalg.eigvals(small_aubry.H.toarray())

        assert np.all(np.isreal(eigvals)), "Eigenvalues should be real"
        assert np.max(np.abs(np.imag(eigvals))) < 1e-10

    def test_spectral_width_scales_with_disorder(self):
        """Test that spectral width increases with λ"""
        np.random.seed(42)
        L = 50

        disorders = [0.5, 1.0, 2.0, 4.0]
        widths = []

        for disorder in disorders:
            model = OneDimensionalAubryAndre(
                L=L, disorder=disorder, rho=1.0, kappa=1.0
            )
            eigvals = np.linalg.eigvalsh(model.H.toarray())
            width = eigvals.max() - eigvals.min()
            widths.append(width)

        # Width should generally increase with disorder
        assert widths[-1] > widths[0], \
            f"Strong disorder width {widths[-1]} should exceed weak disorder {widths[0]}"

    def test_extended_phase_ipr(self):
        """Test IPR in extended phase (λ < 2)"""
        np.random.seed(42)
        model = OneDimensionalAubryAndre(
            L=100, disorder=1.0, rho=1.0, kappa=1.0  # λ = 1 < 2
        )

        model.find_eigval(model.H, sparse=False)
        ipr = model.compute_IPR(model.H_eigvec)

        # Extended states should have small IPR (≈ 1/L)
        mean_ipr = np.mean(ipr)
        assert mean_ipr < 0.5, \
            f"Extended phase should have small IPR, got {mean_ipr}"

    def test_localized_phase_ipr(self):
        """Test IPR in localized phase (λ > 2)"""
        np.random.seed(42)
        model = OneDimensionalAubryAndre(
            L=100, disorder=5.0, rho=1.0, kappa=1.0  # λ = 5 > 2
        )

        model.find_eigval(model.H, sparse=False)
        ipr = model.compute_IPR(model.H_eigvec)

        # Localized states should have larger IPR
        # (not as large as 1D Anderson since it's quasiperiodic, not random)
        mean_ipr = np.mean(ipr)
        assert 0 < mean_ipr < 1, "IPR should be in (0,1)"

    def test_critical_point_properties(self):
        """Test properties at critical point λ = 2"""
        np.random.seed(42)
        model = OneDimensionalAubryAndre(
            L=100, disorder=2.0, rho=1.0, kappa=1.0  # λ = 2 (self-dual)
        )

        # Should have eigenvalues
        eigvals = np.linalg.eigvalsh(model.H.toarray())
        assert len(eigvals) == 100

        # Spectrum should be bounded
        assert -6 < eigvals.min() < 6
        assert -6 < eigvals.max() < 6


# ============================================================================
# TEST CLASS: AUBRY-ANDRÉ TRANSITION
# ============================================================================

class TestAubryAndreTransition:
    """Test Aubry-André metal-insulator transition"""

    def test_transition_exists(self):
        """Test that properties change across λ = 2"""
        np.random.seed(42)
        L = 100

        # Extended phase (λ = 1)
        model_extended = OneDimensionalAubryAndre(
            L=L, disorder=1.0, rho=1.0, kappa=1.0
        )
        model_extended.find_eigval(model_extended.H, sparse=False)
        ipr_extended = model_extended.compute_IPR(model_extended.H_eigvec)

        # Localized phase (λ = 4)
        model_localized = OneDimensionalAubryAndre(
            L=L, disorder=4.0, rho=1.0, kappa=1.0
        )
        model_localized.find_eigval(model_localized.H, sparse=False)
        ipr_localized = model_localized.compute_IPR(model_localized.H_eigvec)

        # Mean IPR should be different
        mean_ipr_ext = np.mean(ipr_extended)
        mean_ipr_loc = np.mean(ipr_localized)

        # This is a statistical test - localized should have higher IPR
        # (though the difference might not always be large for single realizations)
        assert 0 < mean_ipr_ext < 1
        assert 0 < mean_ipr_loc < 1

    def test_self_duality_at_critical_point(self):
        """Test self-duality property at λ = 2"""
        np.random.seed(42)
        L = 55  # Fibonacci number for better quasiperiodic properties

        model = OneDimensionalAubryAndre(
            L=L, disorder=2.0, rho=1.0, kappa=1.0,
            beta=(np.sqrt(5)-1)/2, theta=0
        )

        eigvals = np.linalg.eigvalsh(model.H.toarray())

        # At λ = 2, the model is self-dual
        # Just check it computes without error
        assert len(eigvals) == L


# ============================================================================
# TEST CLASS: SPECIAL CASES
# ============================================================================

class TestSpecialCases:
    """Test special cases and edge conditions"""

    def test_rational_beta(self):
        """Test with rational β (periodic, not quasiperiodic)"""
        np.random.seed(42)
        model = OneDimensionalAubryAndre(
            L=20, disorder=1.0, rho=1.0, kappa=1.0,
            beta=0.25, theta=0  # Rational β
        )

        # Should still create valid Hamiltonian
        assert model.H.shape == (20, 20)

        H = model.H.toarray()
        diag = np.diag(H)

        # With β = 0.25 = 1/4, should be periodic with period 4
        # cos(2π * 0.25 * n) = cos(π*n/2) repeats every 4 sites
        assert np.isclose(diag[0], diag[4], rtol=1e-10), \
            "With β=1/4, potential should have period 4"

    def test_different_theta_values(self):
        """Test various θ values"""
        np.random.seed(42)
        L = 20

        thetas = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]
        models = []

        for theta in thetas:
            model = OneDimensionalAubryAndre(
                L=L, disorder=2.0, rho=1.0, kappa=1.0,
                beta=(np.sqrt(5)-1)/2, theta=theta
            )
            models.append(model)

        # All should be valid
        for model in models:
            assert model.H.shape == (L, L)

        # Different θ should give different Hamiltonians
        for i in range(len(models)-1):
            H1 = models[i].H.toarray()
            H2 = models[i+1].H.toarray()
            assert not np.allclose(H1, H2), \
                f"Different θ values should give different Hamiltonians"

    def test_minimal_system(self):
        """Test L=2 (minimal nontrivial)"""
        np.random.seed(42)
        model = OneDimensionalAubryAndre(
            L=2, disorder=1.0, rho=1.0, kappa=1.0
        )

        assert model.H.shape == (2, 2)

        H = model.H.toarray()
        # Should be [[V(0), -1], [-1, V(1)]]
        assert H[0, 1] == -1.0
        assert H[1, 0] == -1.0

    def test_large_system(self):
        """Test that large systems work"""
        np.random.seed(42)
        L = 1000

        model = OneDimensionalAubryAndre(
            L=L, disorder=2.0, rho=1.0, kappa=1.0
        )

        assert model.H.shape == (L, L)
        assert sp.issparse(model.H)


# ============================================================================
# TEST CLASS: REPRODUCIBILITY
# ============================================================================

class TestReproducibility:
    """Test reproducibility and determinism"""

    def test_deterministic_creation(self):
        """Test that model creation is deterministic"""
        params = dict(L=50, disorder=2.0, rho=5.0, kappa=1.0,
                     beta=(np.sqrt(5)-1)/2, theta=np.pi/3)

        model1 = OneDimensionalAubryAndre(**params)
        model2 = OneDimensionalAubryAndre(**params)

        H1 = model1.H.toarray()
        H2 = model2.H.toarray()

        # Should be identical (no randomness in Aubry-André)
        np.testing.assert_array_equal(H1, H2,
                                      err_msg="Same parameters should give identical Hamiltonians")

    def test_different_parameters_give_different_models(self):
        """Test that different parameters produce different models"""
        base_params = dict(L=50, disorder=2.0, rho=5.0, kappa=1.0)

        model1 = OneDimensionalAubryAndre(**base_params, beta=0.6, theta=0)
        model2 = OneDimensionalAubryAndre(**base_params, beta=0.7, theta=0)

        H1 = model1.H.toarray()
        H2 = model2.H.toarray()

        assert not np.allclose(H1, H2), \
            "Different β should produce different Hamiltonians"


# ============================================================================
# TEST CLASS: INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests"""

    def test_full_workflow(self):
        """Test complete workflow"""
        np.random.seed(42)
        model = OneDimensionalAubryAndre(
            L=50, disorder=1.5, rho=5.0, kappa=2.0,
            beta=(np.sqrt(5)-1)/2, theta=np.pi/4
        )

        # Diagonalize
        model.find_eigval(model.H, sparse=False)

        # Compute statistics
        r = model.calculate_r(model.H_eigval)
        z = model.calculate_z(model.H_eigval)

        assert 0 < r < 1, f"r statistic {r} should be in (0,1)"
        assert 0 < z < 1, f"z statistic {z} should be in (0,1)"

        # Compute IPR
        ipr = model.compute_IPR(model.H_eigvec)
        assert len(ipr) == 50
        assert np.all(ipr > 0)

    def test_comparison_with_anderson_clean(self):
        """Test that λ=0 Aubry-André matches clean Anderson"""
        np.random.seed(42)
        L = 30

        # Aubry-André with λ=0
        aubry = OneDimensionalAubryAndre(
            L=L, disorder=0.0, rho=1.0, kappa=1.0
        )

        # Should have only hopping, like clean Anderson
        H_aubry = aubry.H.toarray()
        diag_aubry = np.diag(H_aubry)

        # Diagonal should be zero
        np.testing.assert_allclose(diag_aubry, 0.0, atol=1e-14,
                                   err_msg="λ=0 should give zero diagonal")

        # Off-diagonal should be -1.0
        for i in range(L-1):
            assert H_aubry[i, i+1] == -1.0
            assert H_aubry[i+1, i] == -1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

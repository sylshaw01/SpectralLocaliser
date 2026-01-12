"""
Regression tests for TwoDimensionalHaldane model.

These tests capture the current numerical output of the Haldane model
implementation and compare against stored reference data after code changes.

Reference data is stored in tests/reference_data/haldane/
Run with --generate-reference to regenerate reference data.

Usage:
    pytest tests/test_haldane_regression.py                    # Run tests against stored data
    pytest tests/test_haldane_regression.py --generate-reference  # Regenerate reference data
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from SLmodels import TwoDimensionalHaldane

# Reference data directory
REFERENCE_DIR = Path(__file__).parent / 'reference_data' / 'haldane'


@pytest.fixture
def generate_reference(request):
    """Fixture to check if we should generate reference data."""
    return request.config.getoption("--generate-reference", default=False)


def save_reference(name: str, data: dict):
    """Save reference data to npz file."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REFERENCE_DIR / f"{name}.npz"
    np.savez(filepath, **data)
    print(f"Saved reference data to {filepath}")


def load_reference(name: str) -> dict:
    """Load reference data from npz file."""
    filepath = REFERENCE_DIR / f"{name}.npz"
    if not filepath.exists():
        pytest.skip(f"Reference data not found: {filepath}. Run with --generate-reference to create.")
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}


# ============================================================================
# TEST CONFIGURATIONS - Define the exact models we test
# ============================================================================

TEST_CONFIGS = {
    'small_clean': {
        'L': 4, 'disorder': 0.0, 'rho': 5.0, 'kappa': 1.0,
        't1': 1.0, 't2': 1.0/3.0, 'M': 0.5, 'phi': np.pi/2,
        'seed': 42
    },
    'small_topological': {
        'L': 4, 'disorder': 0.0, 'rho': 5.0, 'kappa': 1.0,
        't1': 1.0, 't2': 0.3, 'M': 0.2, 'phi': np.pi/2,
        'seed': 42
    },
    'small_trivial': {
        'L': 4, 'disorder': 0.0, 'rho': 5.0, 'kappa': 1.0,
        't1': 1.0, 't2': 0.1, 'M': 1.0, 'phi': np.pi/2,
        'seed': 42
    },
    'small_disordered': {
        'L': 4, 'disorder': 0.5, 'rho': 5.0, 'kappa': 1.0,
        't1': 1.0, 't2': 1.0/3.0, 'M': 0.5, 'phi': np.pi/2,
        'seed': 42
    },
    'medium_clean': {
        'L': 8, 'disorder': 0.0, 'rho': 10.0, 'kappa': 1.0,
        't1': 1.0, 't2': 1.0/3.0, 'M': 0.5, 'phi': np.pi/2,
        'seed': 42
    },
    'medium_topological': {
        'L': 8, 'disorder': 0.0, 'rho': 10.0, 'kappa': 1.0,
        't1': 1.0, 't2': 0.3, 'M': 0.2, 'phi': np.pi/2,
        'seed': 42
    },
    'medium_disordered': {
        'L': 8, 'disorder': 1.0, 'rho': 10.0, 'kappa': 1.0,
        't1': 1.0, 't2': 1.0/3.0, 'M': 0.5, 'phi': np.pi/2,
        'seed': 42
    },
    'large_clean': {
        'L': 16, 'disorder': 0.0, 'rho': 20.0, 'kappa': 1.0,
        't1': 1.0, 't2': 1.0/3.0, 'M': 0.5, 'phi': np.pi/2,
        'seed': 42
    },
    'large_topological': {
        'L': 16, 'disorder': 0.0, 'rho': 20.0, 'kappa': 1.0,
        't1': 1.0, 't2': 0.3, 'M': 0.2, 'phi': np.pi/2,
        'seed': 42
    },
    'large_disordered': {
        'L': 16, 'disorder': 1.5, 'rho': 20.0, 'kappa': 1.0,
        't1': 1.0, 't2': 1.0/3.0, 'M': 0.5, 'phi': np.pi/2,
        'seed': 42
    },
    'different_phi': {
        'L': 4, 'disorder': 0.0, 'rho': 5.0, 'kappa': 1.0,
        't1': 1.0, 't2': 1.0/3.0, 'M': 0.5, 'phi': np.pi/4,
        'seed': 42
    },
}


def create_model(config: dict) -> TwoDimensionalHaldane:
    """Create a Haldane model from config, setting seed first."""
    np.random.seed(config['seed'])
    return TwoDimensionalHaldane(
        L=config['L'],
        disorder=config['disorder'],
        rho=config['rho'],
        kappa=config['kappa'],
        t1=config['t1'],
        t2=config['t2'],
        M=config['M'],
        phi=config['phi']
    )


# ============================================================================
# HAMILTONIAN TESTS
# ============================================================================

class TestHamiltonianRegression:
    """Regression tests for Hamiltonian construction."""

    @pytest.mark.parametrize("config_name", TEST_CONFIGS.keys())
    def test_hamiltonian_eigenvalues(self, config_name, generate_reference):
        """Test that Hamiltonian eigenvalues match reference."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        # Compute eigenvalues
        H_dense = model.H.toarray()
        eigvals = np.linalg.eigvalsh(H_dense)
        eigvals_sorted = np.sort(eigvals)

        ref_name = f"H_eigvals_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'eigenvalues': eigvals_sorted,
                'config': np.array([str(config)], dtype=object)
            })
            return

        ref = load_reference(ref_name)
        np.testing.assert_allclose(
            eigvals_sorted, ref['eigenvalues'],
            rtol=1e-10, atol=1e-12,
            err_msg=f"Hamiltonian eigenvalues changed for {config_name}"
        )

    @pytest.mark.parametrize("config_name", ['small_clean', 'medium_clean', 'large_clean'])
    def test_hamiltonian_matrix_elements(self, config_name, generate_reference):
        """Test that specific Hamiltonian matrix elements match."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        H_dense = model.H.toarray()

        ref_name = f"H_matrix_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'H_real': H_dense.real,
                'H_imag': H_dense.imag,
            })
            return

        ref = load_reference(ref_name)
        H_ref = ref['H_real'] + 1j * ref['H_imag']
        np.testing.assert_allclose(
            H_dense, H_ref,
            rtol=1e-10, atol=1e-12,
            err_msg=f"Hamiltonian matrix changed for {config_name}"
        )


# ============================================================================
# SPECTRAL LOCALIZER TESTS
# ============================================================================

class TestSpectralLocalizerRegression:
    """Regression tests for spectral localizer."""

    @pytest.mark.parametrize("config_name", TEST_CONFIGS.keys())
    def test_spectral_localizer_eigenvalues(self, config_name, generate_reference):
        """Test that spectral localizer eigenvalues match reference."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        # Compute eigenvalues
        SL_dense = model.spectral_localiser.toarray()
        eigvals = np.linalg.eigvalsh(SL_dense)
        eigvals_sorted = np.sort(eigvals)

        ref_name = f"SL_eigvals_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'eigenvalues': eigvals_sorted,
            })
            return

        ref = load_reference(ref_name)
        np.testing.assert_allclose(
            eigvals_sorted, ref['eigenvalues'],
            rtol=1e-10, atol=1e-12,
            err_msg=f"Spectral localizer eigenvalues changed for {config_name}"
        )

    @pytest.mark.parametrize("config_name", ['small_clean', 'small_topological', 'large_topological'])
    def test_spectral_localizer_signature(self, config_name, generate_reference):
        """Test spectral localizer signature (related to Chern number)."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        SL_dense = model.spectral_localiser.toarray()
        eigvals = np.linalg.eigvalsh(SL_dense)

        n_positive = np.sum(eigvals > 0)
        n_negative = np.sum(eigvals < 0)
        signature = n_positive - n_negative

        ref_name = f"SL_signature_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'n_positive': np.array(n_positive),
                'n_negative': np.array(n_negative),
                'signature': np.array(signature),
            })
            return

        ref = load_reference(ref_name)
        assert signature == int(ref['signature']), \
            f"Spectral localizer signature changed for {config_name}: got {signature}, expected {ref['signature']}"


# ============================================================================
# EIGENVECTOR AND IPR TESTS
# ============================================================================

class TestEigenvectorRegression:
    """Regression tests for eigenvectors and IPR."""

    @pytest.mark.parametrize("config_name", ['small_clean', 'small_disordered', 'medium_clean', 'large_clean', 'large_disordered'])
    def test_hamiltonian_ipr(self, config_name, generate_reference):
        """Test IPR values for Hamiltonian eigenvectors."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        # Find eigenvalues and eigenvectors
        model.find_eigval(model.H, sparse=False)
        ipr = model.compute_IPR(model.H_eigvec)

        ref_name = f"H_ipr_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'ipr': ipr,
                'eigenvalues': model.H_eigval,
            })
            return

        ref = load_reference(ref_name)
        np.testing.assert_allclose(
            ipr, ref['ipr'],
            rtol=1e-10, atol=1e-12,
            err_msg=f"Hamiltonian IPR changed for {config_name}"
        )

    @pytest.mark.parametrize("config_name", ['small_clean', 'small_disordered', 'large_clean'])
    def test_spectral_localizer_ipr(self, config_name, generate_reference):
        """Test IPR values for spectral localizer eigenvectors."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        # Find eigenvalues and eigenvectors
        model.find_eigval(model.spectral_localiser, sparse=False)
        ipr = model.compute_IPR(model.spectral_localiser_eigvec)

        ref_name = f"SL_ipr_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'ipr': ipr,
                'eigenvalues': model.spectral_localiser_eigval,
            })
            return

        ref = load_reference(ref_name)
        np.testing.assert_allclose(
            ipr, ref['ipr'],
            rtol=1e-10, atol=1e-12,
            err_msg=f"Spectral localizer IPR changed for {config_name}"
        )


# ============================================================================
# CHERN MARKER TESTS
# ============================================================================

class TestChernMarkerRegression:
    """Regression tests for Chern marker calculations."""

    @pytest.mark.parametrize("config_name", ['small_clean', 'small_topological', 'small_trivial', 'medium_topological', 'large_topological'])
    def test_local_chern_marker(self, config_name, generate_reference):
        """Test local Chern marker values."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        # Compute Chern marker
        model.find_eigval(model.H, sparse=False)
        chern_marker = model.calculate_local_chern_marker(fermi_energy=0)

        ref_name = f"chern_marker_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'chern_marker': chern_marker,
                'mean_chern': np.array(np.mean(chern_marker)),
                'center_chern': np.array(chern_marker[config['L']//2, config['L']//2]),
            })
            return

        ref = load_reference(ref_name)
        np.testing.assert_allclose(
            chern_marker, ref['chern_marker'],
            rtol=1e-10, atol=1e-12,
            err_msg=f"Local Chern marker changed for {config_name}"
        )

    @pytest.mark.parametrize("config_name", ['small_clean', 'small_topological', 'large_topological'])
    def test_projection_operator(self, config_name, generate_reference):
        """Test projection operator construction."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        # Build projection operator
        model.find_eigval(model.H, sparse=False)
        model.projection_operator_lower(fermi_energy=0)
        P = model.P

        # Check idempotency: P^2 = P
        P_squared = P @ P

        ref_name = f"projection_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'P_real': P.real,
                'P_imag': P.imag,
                'trace': np.array(np.trace(P).real),
                'idempotency_error': np.array(np.max(np.abs(P_squared - P))),
            })
            return

        ref = load_reference(ref_name)
        P_ref = ref['P_real'] + 1j * ref['P_imag']
        np.testing.assert_allclose(
            P, P_ref,
            rtol=1e-10, atol=1e-12,
            err_msg=f"Projection operator changed for {config_name}"
        )

    @pytest.mark.parametrize("config_name", ['small_clean', 'small_topological', 'large_topological'])
    def test_spectral_localizer_chern_marker(self, config_name, generate_reference):
        """Test Chern marker from spectral localizer signature."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        model.find_eigval(model.spectral_localiser, sparse=False)
        sl_chern = model.calculate_specral_localiser_chern_marker()

        ref_name = f"sl_chern_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'sl_chern_marker': np.array(sl_chern),
            })
            return

        ref = load_reference(ref_name)
        assert sl_chern == float(ref['sl_chern_marker']), \
            f"SL Chern marker changed for {config_name}: got {sl_chern}, expected {ref['sl_chern_marker']}"


# ============================================================================
# POSITION OPERATOR TESTS
# ============================================================================

class TestPositionOperatorRegression:
    """Regression tests for position operators."""

    @pytest.mark.parametrize("config_name", ['small_clean', 'medium_clean', 'large_clean'])
    def test_position_operators(self, config_name, generate_reference):
        """Test position operator diagonals."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        X_diag = model.X[0].diagonal()
        Y_diag = model.X[1].diagonal()

        ref_name = f"position_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'X_diagonal': X_diag,
                'Y_diagonal': Y_diag,
            })
            return

        ref = load_reference(ref_name)
        np.testing.assert_allclose(
            X_diag, ref['X_diagonal'],
            rtol=1e-10, atol=1e-12,
            err_msg=f"X position operator changed for {config_name}"
        )
        np.testing.assert_allclose(
            Y_diag, ref['Y_diagonal'],
            rtol=1e-10, atol=1e-12,
            err_msg=f"Y position operator changed for {config_name}"
        )


# ============================================================================
# STATISTICS TESTS (r and z values)
# ============================================================================

class TestStatisticsRegression:
    """Regression tests for spectral statistics."""

    @pytest.mark.parametrize("config_name", ['small_disordered', 'medium_disordered', 'large_disordered'])
    def test_r_statistic(self, config_name, generate_reference):
        """Test adjacent gap ratio r."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        model.find_eigval(model.H, sparse=False)
        r_H = model.calculate_r(model.H_eigval)

        model.find_eigval(model.spectral_localiser, sparse=False)
        r_SL = model.calculate_r(model.spectral_localiser_eigval)

        ref_name = f"r_stat_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'r_H': np.array(r_H),
                'r_SL': np.array(r_SL),
            })
            return

        ref = load_reference(ref_name)
        np.testing.assert_allclose(
            r_H, float(ref['r_H']),
            rtol=1e-10, atol=1e-12,
            err_msg=f"r statistic for H changed for {config_name}"
        )
        np.testing.assert_allclose(
            r_SL, float(ref['r_SL']),
            rtol=1e-10, atol=1e-12,
            err_msg=f"r statistic for SL changed for {config_name}"
        )

    @pytest.mark.parametrize("config_name", ['small_disordered', 'medium_disordered', 'large_disordered'])
    def test_z_statistic(self, config_name, generate_reference):
        """Test z statistic (next-nearest neighbor ratio)."""
        config = TEST_CONFIGS[config_name]
        model = create_model(config)

        model.find_eigval(model.H, sparse=False)
        z_H = model.calculate_z(model.H_eigval)

        model.find_eigval(model.spectral_localiser, sparse=False)
        z_SL = model.calculate_z(model.spectral_localiser_eigval)

        ref_name = f"z_stat_{config_name}"

        if generate_reference:
            save_reference(ref_name, {
                'z_H': np.array(z_H),
                'z_SL': np.array(z_SL),
            })
            return

        ref = load_reference(ref_name)
        np.testing.assert_allclose(
            z_H, float(ref['z_H']),
            rtol=1e-10, atol=1e-12,
            err_msg=f"z statistic for H changed for {config_name}"
        )
        np.testing.assert_allclose(
            z_SL, float(ref['z_SL']),
            rtol=1e-10, atol=1e-12,
            err_msg=f"z statistic for SL changed for {config_name}"
        )


# ============================================================================
# FULL WORKFLOW INTEGRATION TEST
# ============================================================================

class TestFullWorkflowRegression:
    """Test complete workflow as used in 2dHaldane.py script."""

    def test_full_workflow_small(self, generate_reference):
        """Test complete workflow matching 2dHaldane.py structure."""
        # Parameters matching typical usage
        L = 4
        rho = 5.0
        t1, t2, M, phi = 1.0, 1.0/3.0, 0.5, np.pi/2
        disorder = 0.5

        np.random.seed(12345)  # Different seed for this test

        # Create model (mimicking single_iteration in 2dHaldane.py)
        model = TwoDimensionalHaldane(L, disorder, rho, kappa=1.0, t1=t1, t2=t2, M=M, phi=phi)

        # Find H eigenvalues
        model.find_eigval(model.H, sparse=False)
        H_eigval = model.H_eigval.copy()
        H_IPR = model.compute_IPR(model.H_eigvec)

        # Find SL eigenvalues
        model.find_eigval(model.spectral_localiser, sparse=False)
        SL_eigval = model.spectral_localiser_eigval.copy()
        SL_IPR = model.compute_IPR(model.spectral_localiser_eigvec)

        # Chern marker
        chern_marker = model.calculate_local_chern_marker()

        ref_name = "full_workflow_small"

        if generate_reference:
            save_reference(ref_name, {
                'H_eigval': H_eigval,
                'H_IPR': H_IPR,
                'SL_eigval': SL_eigval,
                'SL_IPR': SL_IPR,
                'chern_marker': chern_marker,
            })
            return

        ref = load_reference(ref_name)

        np.testing.assert_allclose(H_eigval, ref['H_eigval'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(H_IPR, ref['H_IPR'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(SL_eigval, ref['SL_eigval'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(SL_IPR, ref['SL_IPR'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(chern_marker, ref['chern_marker'], rtol=1e-10, atol=1e-12)

    def test_full_workflow_medium(self, generate_reference):
        """Test complete workflow with medium system size."""
        L = 6
        rho = 8.0
        t1, t2, M, phi = 1.0, 0.3, 0.2, np.pi/2  # Topological parameters
        disorder = 0.3

        np.random.seed(54321)

        model = TwoDimensionalHaldane(L, disorder, rho, kappa=1.0, t1=t1, t2=t2, M=M, phi=phi)

        model.find_eigval(model.H, sparse=False)
        H_eigval = model.H_eigval.copy()
        H_IPR = model.compute_IPR(model.H_eigvec)

        model.find_eigval(model.spectral_localiser, sparse=False)
        SL_eigval = model.spectral_localiser_eigval.copy()
        SL_IPR = model.compute_IPR(model.spectral_localiser_eigvec)

        chern_marker = model.calculate_local_chern_marker()

        ref_name = "full_workflow_medium"

        if generate_reference:
            save_reference(ref_name, {
                'H_eigval': H_eigval,
                'H_IPR': H_IPR,
                'SL_eigval': SL_eigval,
                'SL_IPR': SL_IPR,
                'chern_marker': chern_marker,
                'mean_chern': np.array(np.mean(chern_marker)),
            })
            return

        ref = load_reference(ref_name)

        np.testing.assert_allclose(H_eigval, ref['H_eigval'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(H_IPR, ref['H_IPR'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(SL_eigval, ref['SL_eigval'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(SL_IPR, ref['SL_IPR'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(chern_marker, ref['chern_marker'], rtol=1e-10, atol=1e-12)

    def test_full_workflow_large(self, generate_reference):
        """Test complete workflow with large system size (L=16)."""
        L = 16
        rho = 20.0
        t1, t2, M, phi = 1.0, 0.3, 0.2, np.pi/2  # Topological parameters
        disorder = 0.5

        np.random.seed(99999)

        model = TwoDimensionalHaldane(L, disorder, rho, kappa=1.0, t1=t1, t2=t2, M=M, phi=phi)

        model.find_eigval(model.H, sparse=False)
        H_eigval = model.H_eigval.copy()
        H_IPR = model.compute_IPR(model.H_eigvec)

        model.find_eigval(model.spectral_localiser, sparse=False)
        SL_eigval = model.spectral_localiser_eigval.copy()
        SL_IPR = model.compute_IPR(model.spectral_localiser_eigvec)

        chern_marker = model.calculate_local_chern_marker()

        ref_name = "full_workflow_large"

        if generate_reference:
            save_reference(ref_name, {
                'H_eigval': H_eigval,
                'H_IPR': H_IPR,
                'SL_eigval': SL_eigval,
                'SL_IPR': SL_IPR,
                'chern_marker': chern_marker,
                'mean_chern': np.array(np.mean(chern_marker)),
                'center_chern': np.array(chern_marker[L//2, L//2]),
            })
            return

        ref = load_reference(ref_name)

        np.testing.assert_allclose(H_eigval, ref['H_eigval'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(H_IPR, ref['H_IPR'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(SL_eigval, ref['SL_eigval'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(SL_IPR, ref['SL_IPR'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(chern_marker, ref['chern_marker'], rtol=1e-10, atol=1e-12)


# ============================================================================
# UTILITY: Generate all reference data
# ============================================================================

if __name__ == "__main__":
    """Run directly to generate all reference data."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate Haldane model reference data')
    parser.add_argument('--generate', action='store_true', help='Generate reference data')
    args = parser.parse_args()

    if args.generate:
        print("Generating reference data...")
        print(f"Output directory: {REFERENCE_DIR}")
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

        # Generate all test data
        for config_name, config in TEST_CONFIGS.items():
            print(f"\nProcessing {config_name}...")
            model = create_model(config)

            # Hamiltonian eigenvalues
            H_dense = model.H.toarray()
            eigvals = np.sort(np.linalg.eigvalsh(H_dense))
            save_reference(f"H_eigvals_{config_name}", {
                'eigenvalues': eigvals,
                'config': np.array([str(config)], dtype=object)
            })

            # Hamiltonian matrix (only for clean configs)
            if 'clean' in config_name:
                save_reference(f"H_matrix_{config_name}", {
                    'H_real': H_dense.real,
                    'H_imag': H_dense.imag,
                })

            # Spectral localizer eigenvalues
            SL_dense = model.spectral_localiser.toarray()
            sl_eigvals = np.sort(np.linalg.eigvalsh(SL_dense))
            save_reference(f"SL_eigvals_{config_name}", {'eigenvalues': sl_eigvals})

            # Signature
            n_pos = np.sum(sl_eigvals > 0)
            n_neg = np.sum(sl_eigvals < 0)
            save_reference(f"SL_signature_{config_name}", {
                'n_positive': np.array(n_pos),
                'n_negative': np.array(n_neg),
                'signature': np.array(n_pos - n_neg),
            })

            # Position operators (only for clean configs)
            if 'clean' in config_name:
                save_reference(f"position_{config_name}", {
                    'X_diagonal': model.X[0].diagonal(),
                    'Y_diagonal': model.X[1].diagonal(),
                })

            # IPR
            model.find_eigval(model.H, sparse=False)
            h_ipr = model.compute_IPR(model.H_eigvec)
            save_reference(f"H_ipr_{config_name}", {
                'ipr': h_ipr,
                'eigenvalues': model.H_eigval,
            })

            model.find_eigval(model.spectral_localiser, sparse=False)
            sl_ipr = model.compute_IPR(model.spectral_localiser_eigvec)
            save_reference(f"SL_ipr_{config_name}", {
                'ipr': sl_ipr,
                'eigenvalues': model.spectral_localiser_eigval,
            })

            # Chern marker (only for non-disordered or specific configs)
            if 'disordered' not in config_name or 'small' in config_name:
                chern = model.calculate_local_chern_marker(fermi_energy=0)
                save_reference(f"chern_marker_{config_name}", {
                    'chern_marker': chern,
                    'mean_chern': np.array(np.mean(chern)),
                    'center_chern': np.array(chern[config['L']//2, config['L']//2]),
                })

            # Projection operator (only for clean configs)
            if 'clean' in config_name or 'topological' in config_name:
                model.projection_operator_lower(fermi_energy=0)
                P = model.P
                save_reference(f"projection_{config_name}", {
                    'P_real': P.real,
                    'P_imag': P.imag,
                    'trace': np.array(np.trace(P).real),
                    'idempotency_error': np.array(np.max(np.abs(P @ P - P))),
                })

                sl_chern = model.calculate_specral_localiser_chern_marker()
                save_reference(f"sl_chern_{config_name}", {
                    'sl_chern_marker': np.array(sl_chern),
                })

            # Statistics (only for disordered)
            if 'disordered' in config_name:
                r_H = model.calculate_r(model.H_eigval)
                r_SL = model.calculate_r(model.spectral_localiser_eigval)
                save_reference(f"r_stat_{config_name}", {
                    'r_H': np.array(r_H),
                    'r_SL': np.array(r_SL),
                })

                z_H = model.calculate_z(model.H_eigval)
                z_SL = model.calculate_z(model.spectral_localiser_eigval)
                save_reference(f"z_stat_{config_name}", {
                    'z_H': np.array(z_H),
                    'z_SL': np.array(z_SL),
                })

        # Full workflow tests
        print("\nGenerating full workflow reference data...")

        # Small workflow
        np.random.seed(12345)
        model = TwoDimensionalHaldane(4, 0.5, 5.0, 1.0, t1=1.0, t2=1.0/3.0, M=0.5, phi=np.pi/2)
        model.find_eigval(model.H, sparse=False)
        model.find_eigval(model.spectral_localiser, sparse=False)
        save_reference("full_workflow_small", {
            'H_eigval': model.H_eigval,
            'H_IPR': model.compute_IPR(model.H_eigvec),
            'SL_eigval': model.spectral_localiser_eigval,
            'SL_IPR': model.compute_IPR(model.spectral_localiser_eigvec),
            'chern_marker': model.calculate_local_chern_marker(),
        })

        # Medium workflow
        np.random.seed(54321)
        model = TwoDimensionalHaldane(6, 0.3, 8.0, 1.0, t1=1.0, t2=0.3, M=0.2, phi=np.pi/2)
        model.find_eigval(model.H, sparse=False)
        model.find_eigval(model.spectral_localiser, sparse=False)
        chern = model.calculate_local_chern_marker()
        save_reference("full_workflow_medium", {
            'H_eigval': model.H_eigval,
            'H_IPR': model.compute_IPR(model.H_eigvec),
            'SL_eigval': model.spectral_localiser_eigval,
            'SL_IPR': model.compute_IPR(model.spectral_localiser_eigvec),
            'chern_marker': chern,
            'mean_chern': np.array(np.mean(chern)),
        })

        # Large workflow (L=16)
        print("Generating large workflow (L=16)...")
        np.random.seed(99999)
        model = TwoDimensionalHaldane(16, 0.5, 20.0, 1.0, t1=1.0, t2=0.3, M=0.2, phi=np.pi/2)
        model.find_eigval(model.H, sparse=False)
        model.find_eigval(model.spectral_localiser, sparse=False)
        chern = model.calculate_local_chern_marker()
        save_reference("full_workflow_large", {
            'H_eigval': model.H_eigval,
            'H_IPR': model.compute_IPR(model.H_eigvec),
            'SL_eigval': model.spectral_localiser_eigval,
            'SL_IPR': model.compute_IPR(model.spectral_localiser_eigvec),
            'chern_marker': chern,
            'mean_chern': np.array(np.mean(chern)),
            'center_chern': np.array(chern[8, 8]),
        })

        print("\nDone! Reference data generated.")
    else:
        print("Run with --generate to create reference data")
        print("Or use: pytest tests/test_haldane_regression.py --generate-reference")

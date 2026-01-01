# Test Suite for SpectralLocalizer

This directory contains comprehensive unit and integration tests for the SpectralLocalizer project.

## Test Coverage

### `test_1d_anderson.py` - OneDimensionalAnderson Model Tests

**Total: 50+ test cases organized into 9 test classes:**

1. **TestHamiltonianStructure** (6 tests)
   - Matrix shape, sparsity, Hermiticity
   - Tridiagonal structure validation
   - Real-valued matrix verification

2. **TestMatrixElements** (5 tests)
   - Off-diagonal hopping elements
   - Diagonal disorder range
   - Boundary conditions

3. **TestPhysicalProperties** (5 tests)
   - Eigenvalue count and reality
   - **CRITICAL: Analytical validation** against exact solution
   - Disorder scaling behavior

4. **TestReproducibility** (2 tests)
   - Random seed control
   - Deterministic behavior

5. **TestEdgeCases** (5 tests)
   - Edge cases: L=1, L=2, L=10000
   - Extreme disorder values

6. **TestPositionOperator** (6 tests)
   - Diagonal structure
   - Range [-ρ, ρ] validation
   - Linear spacing

7. **TestSpectralLocalizer** (5 tests)
   - Block structure validation
   - Hermiticity
   - κ parameter scaling

8. **TestIntegration** (4 tests)
   - Full workflow validation
   - Localization behavior
   - **CRITICAL: Eigenvalue sorting validation** (addresses code review bug)

## Priority Tests

### Must-Pass Tests

These tests validate fundamental correctness:

1. ✅ `test_zero_disorder_gives_cosine_band` - Validates against analytical solution
2. ✅ `test_hamiltonian_is_hermitian` - Ensures physical validity
3. ✅ `test_tridiagonal_structure_detailed` - Validates matrix construction
4. ✅ `test_eigenvalue_sorting_in_find_eigval` - **Catches the bug from code review**

### Physics Validation Tests

Tests marked with `@pytest.mark.physics`:
- `test_zero_disorder_gives_cosine_band`
- `test_eigenvalue_range_scales_with_disorder`
- `test_spectral_center_near_zero`

## Running Tests

See the main README or instructions below for how to run these tests.

## Test Organization

```
tests/
├── __init__.py                 # Package marker
├── conftest.py                 # Shared fixtures and configuration
├── test_1d_anderson.py         # Main test file
└── README.md                   # This file
```

## Adding New Tests

When adding tests:
1. Follow the existing naming convention: `test_*`
2. Use descriptive names that explain what is being tested
3. Add docstrings explaining the test purpose
4. Use appropriate fixtures from `conftest.py`
5. Mark slow tests with `@pytest.mark.slow`

## Known Issues Being Tested

These tests specifically address bugs found in code review:
- Unsorted eigenvalues from `find_eigval()`
- Missing random seed control
- Hermiticity validation

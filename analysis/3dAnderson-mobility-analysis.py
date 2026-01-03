"""
3D Anderson Model Analysis Script
=================================
Investigates the disorder-driven metal-insulator transition in the 3D Anderson model,
with focus on the mobility edge and spectral localizer comparison.

Generates:
- Figure 1: DOS and IPR summary (2x3 grid per disorder value)
- Figure 2: Energy-resolved IPR (mobility edge visualization)
- Figure 3: Mobility edge trajectory vs disorder
- Figure 4: Filtered r/z statistics (using H's mobility edge)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob
import argparse
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default data directory (relative to this script)
DATA_DIR = '../data/'
FIGURE_DIR = '../figures/'

# Set to True to compute IPR from eigenvectors, False to skip IPR-dependent plots
COMPUTE_IPR = True

# Disorder indices to plot (None = auto-select low/mid/high)
# Examples: [0, 2, 5] for specific indices, or None for automatic selection
DISORDER_INDICES = None

# IPR percentile cutoff for energy-resolved plots (removes outliers)
# e.g., 90 means show only the bottom 90% of IPR values
IPR_PERCENTILE_CUTOFF = 90

# Plot style constants (matching notebooks/recentdata.ipynb)
COLORS = {'H': 'blue', 'SL': 'orange'}
FIGSIZE_2x3 = (18, 10)
FIGSIZE_2x2 = (18, 18)
TITLE_SIZE = 20
LABEL_SIZE = 20
SUPTITLE_SIZE = 24
ANNOTATION_PROPS = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# Reference values for spectral statistics
GOE_R = 0.5295
POISSON_R = 0.386
GOE_Z = 0.5687
POISSON_Z = 0.5

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def parse_filename(filepath):
    """
    Parse a 3D Anderson data filename to extract parameters.

    Expected format: 3dAnderson_L{L}_disorder{start}-{end}_numEigs{n}_realizations{r}_{date}_*.dat

    Returns dict with keys: L, disorder_start, disorder_end, num_eigs, num_realizations, date
    """
    basename = os.path.basename(filepath)

    params = {}

    # Extract L
    match = re.search(r'_L(\d+)_', basename)
    if match:
        params['L'] = int(match.group(1))

    # Extract disorder range
    match = re.search(r'_disorder([\d.]+)-([\d.]+)_', basename)
    if match:
        params['disorder_start'] = float(match.group(1))
        params['disorder_end'] = float(match.group(2))

    # Extract number of eigenvalues
    match = re.search(r'_numEigs(\d+)_', basename)
    if match:
        params['num_eigs'] = int(match.group(1))

    # Extract number of realizations
    match = re.search(r'_realizations(\d+)_', basename)
    if match:
        params['num_realizations'] = int(match.group(1))

    # Extract date
    match = re.search(r'_(\d{4}-\d{2}-\d{2})-(\d+)_', basename)
    if match:
        params['date'] = match.group(1)

    return params


def find_data_files(data_dir, L=None):
    """
    Find all 3D Anderson data files in the given directory.

    If L is specified, only return files for that system size.

    Returns dict mapping L values to dict of file paths by type.
    """
    pattern = os.path.join(data_dir, '3dAnderson_L*_disorder*_*.dat')
    all_files = glob.glob(pattern)

    # Group files by L and type
    data_files = {}

    for filepath in all_files:
        params = parse_filename(filepath)
        if 'L' not in params:
            continue

        file_L = params['L']
        if L is not None and file_L != L:
            continue

        if file_L not in data_files:
            data_files[file_L] = {'params': params, 'files': {}}

        # Determine file type from suffix
        basename = os.path.basename(filepath)
        if basename.endswith('_H_eigval.dat'):
            data_files[file_L]['files']['H_eigval'] = filepath
        elif basename.endswith('_H_eigvec.dat'):
            data_files[file_L]['files']['H_eigvec'] = filepath
        elif basename.endswith('_H_IPR.dat'):
            data_files[file_L]['files']['H_IPR'] = filepath  # May not be used if incorrect
        elif basename.endswith('_spectral_localiser_eigval.dat'):
            data_files[file_L]['files']['SL_eigval'] = filepath
        elif basename.endswith('_spectral_localiser_eigvec.dat'):
            data_files[file_L]['files']['SL_eigvec'] = filepath
        elif basename.endswith('_spectral_localiser_IPR.dat'):
            data_files[file_L]['files']['SL_IPR'] = filepath  # May not be used if incorrect
        elif basename.endswith('_parameters.txt'):
            data_files[file_L]['files']['parameters'] = filepath
        elif basename.endswith('_seeds.dat'):
            data_files[file_L]['files']['seeds'] = filepath

    return data_files


def load_parameters(param_file):
    """Load parameters from a parameters.txt file."""
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()
    return params


def compute_ipr(eigvec):
    """
    Compute the Inverse Participation Ratio (IPR) for eigenvectors.

    IPR = sum(|psi_i|^4)

    For a fully delocalized state: IPR ~ 1/N
    For a fully localized state: IPR ~ 1

    Args:
        eigvec: array of shape (..., N) where last dimension is the eigenvector components
                or shape (..., N, M) where M is number of eigenvectors

    Returns:
        IPR values with shape matching input minus the eigenvector dimension
    """
    # IPR = sum of |psi|^4
    return np.sum(np.abs(eigvec) ** 4, axis=-1)


def load_data(data_files_dict, L):
    """
    Load all data for a given system size L.

    Returns dict with:
        - H_eigval: array of shape (num_disorder, num_realizations, num_eigs)
        - H_eigvec: array of shape (num_disorder, num_realizations, L^3, num_eigs) [if available]
        - SL_eigval: array of shape (num_disorder, num_realizations, num_eigs_sl)
        - SL_eigvec: array of shape (num_disorder, num_realizations, 4*L^3, num_eigs_sl) [if available]
        - disorder_values: 1D array of disorder strengths
        - params: dict of parameters
        - has_eigvec: bool indicating if eigenvectors are available
    """
    if L not in data_files_dict:
        raise ValueError(f"No data found for L={L}")

    file_info = data_files_dict[L]
    files = file_info['files']
    params = file_info['params']

    # Load parameters file if available
    full_params = {}
    if 'parameters' in files:
        full_params = load_parameters(files['parameters'])
        # Get disorder resolution
        disorder_resolution = int(full_params.get('disorder_resolution', 6))
    else:
        # Infer from filename or use default
        disorder_resolution = 6  # Default, will be corrected below

    # Get disorder range and realizations - prefer full_params (from parameters.txt),
    # then params (from filename), then defaults
    disorder_start = float(full_params.get('disorder_start', params.get('disorder_start', 0.0)))
    disorder_end = float(full_params.get('disorder_end', params.get('disorder_end', 30.0)))
    num_realizations = int(full_params.get('num_disorder_realisations',
                          params.get('num_realizations', 5)))
    num_eigs = params.get('num_eigs', L**3)  # Default to full spectrum

    # Calculate sizes
    H_size = L ** 3
    SL_size = 4 * L ** 3

    # Try to infer disorder_resolution and num_eigs from file size
    if 'H_eigval' in files:
        file_size = os.path.getsize(files['H_eigval'])
        # float64 = 8 bytes, shape is (disorder_resolution, num_realizations, num_eigs)
        total_elements = file_size // 8
        # We know num_realizations, so: total = disorder_resolution * num_realizations * num_eigs
        # For full spectrum: num_eigs = H_size = L^3
        # Try to infer both
        if total_elements % (num_realizations * H_size) == 0:
            disorder_resolution = total_elements // (num_realizations * H_size)
            num_eigs_H = H_size
        else:
            # num_eigs might be different from H_size (sparse solver case)
            # Try common values
            for dr in range(1, 100):
                if total_elements % (dr * num_realizations) == 0:
                    possible_num_eigs = total_elements // (dr * num_realizations)
                    if possible_num_eigs <= H_size:
                        disorder_resolution = dr
                        num_eigs_H = possible_num_eigs
                        break
            else:
                num_eigs_H = H_size
    else:
        num_eigs_H = H_size

    # Infer SL num_eigs similarly
    if 'SL_eigval' in files:
        file_size = os.path.getsize(files['SL_eigval'])
        total_elements = file_size // 8
        num_eigs_SL = total_elements // (disorder_resolution * num_realizations)
    else:
        num_eigs_SL = SL_size

    shape_H_eigval = (disorder_resolution, num_realizations, num_eigs_H)
    shape_SL_eigval = (disorder_resolution, num_realizations, num_eigs_SL)

    # Eigenvector shapes: (disorder, realizations, basis_size, num_eigs)
    shape_H_eigvec = (disorder_resolution, num_realizations, H_size, H_size)
    shape_SL_eigvec = (disorder_resolution, num_realizations, SL_size, SL_size)

    data = {
        'params': params,
        'disorder_values': np.linspace(disorder_start, disorder_end, disorder_resolution),
        'L': L,
        'has_eigvec': False
    }

    # Load memmap arrays for eigenvalues
    if 'H_eigval' in files:
        data['H_eigval'] = np.memmap(files['H_eigval'], dtype='float64', mode='r', shape=shape_H_eigval)
        print(f"  Loaded H_eigval: shape {shape_H_eigval}")

    if 'SL_eigval' in files:
        data['SL_eigval'] = np.memmap(files['SL_eigval'], dtype='float64', mode='r', shape=shape_SL_eigval)
        print(f"  Loaded SL_eigval: shape {shape_SL_eigval}")

    # Load eigenvectors if available (needed for IPR calculation)
    if 'H_eigvec' in files:
        data['H_eigvec'] = np.memmap(files['H_eigvec'], dtype='complex128', mode='r', shape=shape_H_eigvec)
        data['has_eigvec'] = True
        print(f"  Loaded H_eigvec: shape {shape_H_eigvec}")

    if 'SL_eigvec' in files:
        data['SL_eigvec'] = np.memmap(files['SL_eigvec'], dtype='complex128', mode='r', shape=shape_SL_eigvec)
        print(f"  Loaded SL_eigvec: shape {shape_SL_eigvec}")

    # Note: IPR will be computed on-the-fly from eigenvectors, not loaded from files
    # The pre-computed IPR files were not stored correctly

    return data


# ============================================================================
# SPECTRAL STATISTICS
# ============================================================================

def calculate_r(eigval):
    """
    Calculate the adjacent gap ratio r for a sorted array of eigenvalues.
    r = min(s_i, s_{i+1}) / max(s_i, s_{i+1})

    Returns mean r value.
    """
    eigval_sorted = np.sort(eigval)
    spacings = np.diff(eigval_sorted)

    min_vals = np.minimum(spacings[:-1], spacings[1:])
    max_vals = np.maximum(spacings[:-1], spacings[1:])

    r = np.divide(min_vals, max_vals, out=np.zeros_like(min_vals), where=max_vals != 0)
    return r.mean()


def calculate_z(eigval):
    """
    Calculate the next-nearest neighbor ratio z.

    Returns mean z value.
    """
    eigval_sorted = np.sort(eigval)
    s = np.diff(eigval_sorted)

    if len(s) < 5:
        return np.nan

    s_i_minus_2 = s[:-4]
    s_i_minus_1 = s[1:-3]
    s_i = s[2:-2]
    s_i_plus_1 = s[3:-1]

    nn = np.minimum(s_i, s_i_minus_1)
    n_other = np.maximum(s_i, s_i_minus_1)
    nnn_left = s_i_minus_1 + s_i_minus_2
    nnn_right = s_i + s_i_plus_1

    nnn = np.minimum.reduce([n_other, nnn_left, nnn_right])

    z = np.divide(nn, nnn, out=np.zeros_like(nn), where=nnn != 0)
    return z.mean()


def filter_eigenvalues_by_energy(eigval, E_min, E_max):
    """Filter eigenvalues to those within the energy window [E_min, E_max]."""
    mask = (eigval >= E_min) & (eigval <= E_max)
    return eigval[mask]


# ============================================================================
# MOBILITY EDGE EXTRACTION
# ============================================================================

def extract_mobility_edge(eigval, ipr, num_bins=50, ipr_threshold=None):
    """
    Extract the mobility edge from energy-resolved IPR data.

    The mobility edge is defined as the energy where IPR exceeds a threshold.
    If no threshold is given, uses 2/N where N is the number of eigenvalues.

    Args:
        eigval: 1D array of eigenvalues
        ipr: 1D array of corresponding IPR values
        num_bins: number of energy bins
        ipr_threshold: IPR threshold for localization (default: 2/N)

    Returns:
        E_c_lower: lower mobility edge (negative energy side)
        E_c_upper: upper mobility edge (positive energy side)
        bin_centers: energy bin centers
        bin_ipr: mean IPR in each bin
    """
    if ipr_threshold is None:
        ipr_threshold = 2.0 / len(eigval)

    # Create energy bins
    E_min, E_max = eigval.min(), eigval.max()
    bin_edges = np.linspace(E_min, E_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compute mean IPR in each bin
    bin_ipr = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    bin_indices = np.digitize(eigval, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    for i in range(len(eigval)):
        bin_idx = bin_indices[i]
        bin_ipr[bin_idx] += ipr[i]
        bin_counts[bin_idx] += 1

    # Avoid division by zero
    bin_ipr = np.divide(bin_ipr, bin_counts, out=np.zeros_like(bin_ipr), where=bin_counts > 0)

    # Find mobility edges (where IPR crosses threshold)
    # Lower edge: find the highest negative energy where IPR > threshold
    # Upper edge: find the lowest positive energy where IPR > threshold

    center_idx = num_bins // 2

    # Search outward from center for localized regions
    E_c_lower = E_min  # Default to band edge
    E_c_upper = E_max

    # Lower mobility edge (search from center toward negative energies)
    for i in range(center_idx, -1, -1):
        if bin_counts[i] > 0 and bin_ipr[i] > ipr_threshold:
            E_c_lower = bin_centers[i]
            break

    # Upper mobility edge (search from center toward positive energies)
    for i in range(center_idx, num_bins):
        if bin_counts[i] > 0 and bin_ipr[i] > ipr_threshold:
            E_c_upper = bin_centers[i]
            break

    return E_c_lower, E_c_upper, bin_centers, bin_ipr


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_dos_ipr_summary(data, disorder_idx, realization_idx=0, save_path=None):
    """
    Plot DOS and IPR summary for a single disorder value (Figure 1 style).

    Creates a 2x3 grid (or 2x2 if IPR not available):
    - Col 0: DOS (horizontal histogram)
    - Col 1: Eigenvalues vs Index
    - Col 2: IPR vs Index (only if COMPUTE_IPR=True)
    - Row 0: Hamiltonian
    - Row 1: Spectral Localizer
    """
    L = data['L']
    W = data['disorder_values'][disorder_idx]

    # Get data for this disorder and realization
    H_eigval = data['H_eigval'][disorder_idx, realization_idx]
    SL_eigval = data['SL_eigval'][disorder_idx, realization_idx]

    # Check if we should compute IPR
    compute_ipr_flag = COMPUTE_IPR and data.get('has_eigvec', False)

    if compute_ipr_flag:
        fig, axs = plt.subplots(2, 3, figsize=FIGSIZE_2x3, constrained_layout=True)

        # Get eigenvectors and compute IPR on-the-fly
        H_eigvec = data['H_eigvec'][disorder_idx, realization_idx]
        SL_eigvec = data['SL_eigvec'][disorder_idx, realization_idx]
        H_IPR = compute_ipr(H_eigvec.T)
        SL_IPR = compute_ipr(SL_eigvec.T)
    else:
        # 2x2 grid without IPR
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        H_IPR = None
        SL_IPR = None

    # Column 0: DOS (horizontal histograms)
    axs[0, 0].hist(H_eigval, bins=100, density=True, orientation='horizontal',
                   color=COLORS['H'], alpha=0.8)
    axs[0, 0].set_title('Hamiltonian DOS', size=TITLE_SIZE)
    axs[0, 0].set_xlabel('P(E)', size=LABEL_SIZE)
    axs[0, 0].set_ylabel('Energy (E)', size=LABEL_SIZE)
    axs[0, 0].grid(True)
    axs[0, 0].set_axisbelow(True)

    axs[1, 0].hist(SL_eigval, bins=100, density=True, orientation='horizontal',
                   color=COLORS['SL'], alpha=0.8)
    axs[1, 0].set_title('Spectral Localiser DOS', size=TITLE_SIZE)
    axs[1, 0].set_xlabel('P(E)', size=LABEL_SIZE)
    axs[1, 0].set_ylabel('Eigenvalue', size=LABEL_SIZE)
    axs[1, 0].grid(True)
    axs[1, 0].set_axisbelow(True)

    # Column 1: Eigenvalues vs Index
    H_indices = np.arange(len(H_eigval))
    SL_indices = np.arange(len(SL_eigval))

    axs[0, 1].scatter(H_indices, np.sort(H_eigval), s=1, c=COLORS['H'], alpha=0.5)
    axs[0, 1].set_title('Hamiltonian Eigenvalues', size=TITLE_SIZE)
    axs[0, 1].set_xlabel('Index', size=LABEL_SIZE)
    axs[0, 1].set_ylabel('Energy (E)', size=LABEL_SIZE)
    axs[0, 1].grid(True)
    axs[0, 1].set_axisbelow(True)

    axs[1, 1].scatter(SL_indices, np.sort(SL_eigval), s=1, c=COLORS['SL'], alpha=0.5)
    axs[1, 1].set_title('Spectral Localiser Eigenvalues', size=TITLE_SIZE)
    axs[1, 1].set_xlabel('Index', size=LABEL_SIZE)
    axs[1, 1].set_ylabel('Eigenvalue', size=LABEL_SIZE)
    axs[1, 1].grid(True)
    axs[1, 1].set_axisbelow(True)

    # Column 2: IPR vs Index (only if computing IPR)
    if compute_ipr_flag and H_IPR is not None:
        # Sort eigenvalues and reorder IPR accordingly
        H_sort_idx = np.argsort(H_eigval)
        SL_sort_idx = np.argsort(SL_eigval)

        axs[0, 2].scatter(H_indices, H_IPR[H_sort_idx], s=1, c=COLORS['H'], alpha=0.5)
        axs[0, 2].set_title('Hamiltonian IPR', size=TITLE_SIZE)
        axs[0, 2].set_xlabel('Index', size=LABEL_SIZE)
        axs[0, 2].set_ylabel('IPR', size=LABEL_SIZE)
        axs[0, 2].grid(True)
        axs[0, 2].set_axisbelow(True)

        # Add average IPR annotation
        avg_H_IPR = np.mean(H_IPR)
        axs[0, 2].text(0.95, 0.95, f'ave={avg_H_IPR:.4f}', transform=axs[0, 2].transAxes,
                       fontsize=14, verticalalignment='top', horizontalalignment='right',
                       bbox=ANNOTATION_PROPS)

        axs[1, 2].scatter(SL_indices, SL_IPR[SL_sort_idx], s=1, c=COLORS['SL'], alpha=0.5)
        axs[1, 2].set_title('Spectral Localiser IPR', size=TITLE_SIZE)
        axs[1, 2].set_xlabel('Index', size=LABEL_SIZE)
        axs[1, 2].set_ylabel('IPR', size=LABEL_SIZE)
        axs[1, 2].grid(True)
        axs[1, 2].set_axisbelow(True)

        avg_SL_IPR = np.mean(SL_IPR)
        axs[1, 2].text(0.95, 0.95, f'ave={avg_SL_IPR:.4f}', transform=axs[1, 2].transAxes,
                       fontsize=14, verticalalignment='top', horizontalalignment='right',
                       bbox=ANNOTATION_PROPS)

        fig.suptitle(f'DOS and IPR at Disorder W={W:.2f}, L={L}', fontsize=SUPTITLE_SIZE)
    else:
        fig.suptitle(f'DOS and Eigenvalues at Disorder W={W:.2f}, L={L}', fontsize=SUPTITLE_SIZE)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axs


def plot_energy_resolved_ipr(data, disorder_indices, save_path=None):
    """
    Plot energy-resolved IPR to visualize the mobility edge (Figure 2).

    Creates a 2xN grid showing IPR vs Energy for different disorder values.
    """
    if not COMPUTE_IPR:
        print("Skipping energy-resolved IPR plot (COMPUTE_IPR=False)")
        return None, None

    if not data.get('has_eigvec', False):
        print("Warning: Eigenvector data not available. Cannot compute IPR.")
        print("Make sure eigenvector files (*_eigvec.dat) are present.")
        return None, None

    n_disorders = len(disorder_indices)
    fig, axs = plt.subplots(2, n_disorders, figsize=(6*n_disorders, 10), constrained_layout=True)

    if n_disorders == 1:
        axs = axs.reshape(2, 1)

    L = data['L']
    n_realizations = data['H_eigval'].shape[1]

    for col, d_idx in enumerate(disorder_indices):
        W = data['disorder_values'][d_idx]

        # Aggregate over realizations, computing IPR on-the-fly
        H_eigval_list = []
        H_IPR_list = []
        SL_eigval_list = []
        SL_IPR_list = []

        for r_idx in range(n_realizations):
            H_eigval_list.append(data['H_eigval'][d_idx, r_idx])
            SL_eigval_list.append(data['SL_eigval'][d_idx, r_idx])

            # Compute IPR from eigenvectors
            H_eigvec = data['H_eigvec'][d_idx, r_idx]
            SL_eigvec = data['SL_eigvec'][d_idx, r_idx]
            H_IPR_list.append(compute_ipr(H_eigvec.T))
            SL_IPR_list.append(compute_ipr(SL_eigvec.T))

        H_eigval = np.concatenate(H_eigval_list)
        H_IPR = np.concatenate(H_IPR_list)
        SL_eigval = np.concatenate(SL_eigval_list)
        SL_IPR = np.concatenate(SL_IPR_list)

        # Apply percentile cutoff to remove outliers
        H_ipr_cutoff = np.percentile(H_IPR, IPR_PERCENTILE_CUTOFF)
        SL_ipr_cutoff = np.percentile(SL_IPR, IPR_PERCENTILE_CUTOFF)

        H_mask = H_IPR <= H_ipr_cutoff
        SL_mask = SL_IPR <= SL_ipr_cutoff

        # Hamiltonian: Energy vs IPR (flipped axes)
        axs[0, col].scatter(H_eigval[H_mask], H_IPR[H_mask], s=0.5, c=COLORS['H'], alpha=0.3)
        axs[0, col].set_title(f'H: W={W:.1f}', size=TITLE_SIZE)
        axs[0, col].set_xlabel('Energy (E)', size=LABEL_SIZE)
        axs[0, col].set_ylabel('IPR', size=LABEL_SIZE)
        axs[0, col].grid(True)
        axs[0, col].set_axisbelow(True)

        # Spectral Localizer: Energy vs IPR (flipped axes)
        axs[1, col].scatter(SL_eigval[SL_mask], SL_IPR[SL_mask], s=0.5, c=COLORS['SL'], alpha=0.3)
        axs[1, col].set_title(f'SL: W={W:.1f}', size=TITLE_SIZE)
        axs[1, col].set_xlabel('Eigenvalue', size=LABEL_SIZE)
        axs[1, col].set_ylabel('IPR', size=LABEL_SIZE)
        axs[1, col].grid(True)
        axs[1, col].set_axisbelow(True)

    fig.suptitle(f'Energy-Resolved IPR (bottom {IPR_PERCENTILE_CUTOFF}%), L={L}', fontsize=SUPTITLE_SIZE)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axs


def plot_mobility_edge_trajectory(all_data, ipr_threshold=None, save_path=None):
    """
    Plot mobility edge trajectory vs disorder for different system sizes (Figure 3).
    """
    if not COMPUTE_IPR:
        print("Skipping mobility edge trajectory plot (COMPUTE_IPR=False)")
        return None, None

    # Check if any data has eigenvectors
    has_any_eigvec = any(d.get('has_eigvec', False) for d in all_data.values())
    if not has_any_eigvec:
        print("Warning: No eigenvector data available. Cannot compute mobility edge.")
        return None, None

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    colors_L = plt.cm.viridis(np.linspace(0.2, 0.8, len(all_data)))

    for (L, data), color in zip(sorted(all_data.items()), colors_L):
        if not data.get('has_eigvec', False):
            print(f"Skipping L={L}: no eigenvector data")
            continue

        disorder_values = data['disorder_values']
        n_disorder = len(disorder_values)
        n_realizations = data['H_eigval'].shape[1]

        H_Ec_lower = np.zeros((n_disorder, n_realizations))
        H_Ec_upper = np.zeros((n_disorder, n_realizations))

        for d_idx in range(n_disorder):
            for r_idx in range(n_realizations):
                H_eigval = data['H_eigval'][d_idx, r_idx]
                # Compute IPR from eigenvectors
                H_eigvec = data['H_eigvec'][d_idx, r_idx]
                H_IPR = compute_ipr(H_eigvec.T)

                threshold = ipr_threshold if ipr_threshold else 2.0 / len(H_eigval)
                Ec_l, Ec_u, _, _ = extract_mobility_edge(H_eigval, H_IPR, ipr_threshold=threshold)
                H_Ec_lower[d_idx, r_idx] = Ec_l
                H_Ec_upper[d_idx, r_idx] = Ec_u

        # Mean and std across realizations
        Ec_lower_mean = H_Ec_lower.mean(axis=1)
        Ec_lower_std = H_Ec_lower.std(axis=1)
        Ec_upper_mean = H_Ec_upper.mean(axis=1)
        Ec_upper_std = H_Ec_upper.std(axis=1)

        # Plot lower mobility edge
        axs[0].errorbar(disorder_values, Ec_lower_mean, yerr=Ec_lower_std,
                        label=f'L={L}', color=color, marker='o', capsize=3)

        # Plot upper mobility edge
        axs[1].errorbar(disorder_values, Ec_upper_mean, yerr=Ec_upper_std,
                        label=f'L={L}', color=color, marker='o', capsize=3)

    axs[0].set_title('Lower Mobility Edge (E < 0)', size=TITLE_SIZE)
    axs[0].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[0].set_ylabel('E_c (lower)', size=LABEL_SIZE)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Upper Mobility Edge (E > 0)', size=TITLE_SIZE)
    axs[1].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[1].set_ylabel('E_c (upper)', size=LABEL_SIZE)
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle('Mobility Edge Trajectory vs Disorder', fontsize=SUPTITLE_SIZE)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axs


def plot_filtered_rz_statistics(data, ipr_threshold=None, save_path=None):
    """
    Plot r and z statistics filtered by the mobility edge energy window (Figure 4).

    Uses H's mobility edge E_c to filter both H and SL eigenvalues.
    Requires COMPUTE_IPR=True to determine mobility edge.
    """
    if not COMPUTE_IPR:
        print("Skipping filtered r/z statistics plot (COMPUTE_IPR=False)")
        print("Note: This plot requires mobility edge from IPR. Consider plot_rz_statistics_unfiltered() instead.")
        return None, None

    if not data.get('has_eigvec', False):
        print("Warning: Eigenvector data not available. Cannot compute IPR for mobility edge.")
        print("Make sure eigenvector files (*_eigvec.dat) are present.")
        return None, None

    fig, axs = plt.subplots(2, 2, figsize=FIGSIZE_2x2, constrained_layout=True)

    L = data['L']
    disorder_values = data['disorder_values']
    n_disorder = len(disorder_values)
    n_realizations = data['H_eigval'].shape[1]

    # Arrays to store results
    H_r_vals = np.zeros((n_disorder, n_realizations))
    H_z_vals = np.zeros((n_disorder, n_realizations))
    SL_r_vals = np.zeros((n_disorder, n_realizations))
    SL_z_vals = np.zeros((n_disorder, n_realizations))

    for d_idx in range(n_disorder):
        for r_idx in range(n_realizations):
            H_eigval = data['H_eigval'][d_idx, r_idx]
            SL_eigval = data['SL_eigval'][d_idx, r_idx]

            # Compute IPR from eigenvectors
            H_eigvec = data['H_eigvec'][d_idx, r_idx]
            H_IPR = compute_ipr(H_eigvec.T)

            # Extract mobility edge from H
            threshold = ipr_threshold if ipr_threshold else 2.0 / len(H_eigval)
            Ec_l, Ec_u, _, _ = extract_mobility_edge(H_eigval, H_IPR, ipr_threshold=threshold)

            # Filter H eigenvalues by mobility edge
            H_filtered = filter_eigenvalues_by_energy(H_eigval, Ec_l, Ec_u)

            # Filter SL eigenvalues using H's mobility edge
            SL_filtered = filter_eigenvalues_by_energy(SL_eigval, Ec_l, Ec_u)

            # Calculate r and z for filtered eigenvalues
            if len(H_filtered) > 5:
                H_r_vals[d_idx, r_idx] = calculate_r(H_filtered)
                H_z_vals[d_idx, r_idx] = calculate_z(H_filtered)
            else:
                H_r_vals[d_idx, r_idx] = np.nan
                H_z_vals[d_idx, r_idx] = np.nan

            if len(SL_filtered) > 5:
                SL_r_vals[d_idx, r_idx] = calculate_r(SL_filtered)
                SL_z_vals[d_idx, r_idx] = calculate_z(SL_filtered)
            else:
                SL_r_vals[d_idx, r_idx] = np.nan
                SL_z_vals[d_idx, r_idx] = np.nan

    # Compute means and standard errors
    H_r_mean = np.nanmean(H_r_vals, axis=1)
    H_r_std = np.nanstd(H_r_vals, axis=1) / np.sqrt(n_realizations)
    H_z_mean = np.nanmean(H_z_vals, axis=1)
    H_z_std = np.nanstd(H_z_vals, axis=1) / np.sqrt(n_realizations)

    SL_r_mean = np.nanmean(SL_r_vals, axis=1)
    SL_r_std = np.nanstd(SL_r_vals, axis=1) / np.sqrt(n_realizations)
    SL_z_mean = np.nanmean(SL_z_vals, axis=1)
    SL_z_std = np.nanstd(SL_z_vals, axis=1) / np.sqrt(n_realizations)

    # Plot H r-statistic
    axs[0, 0].errorbar(disorder_values, H_r_mean, yerr=H_r_std,
                        label=f'L={L}', marker='o', capsize=3, color=COLORS['H'])
    axs[0, 0].axhline(y=GOE_R, color='red', linestyle='--', label='GOE', alpha=0.7)
    axs[0, 0].axhline(y=POISSON_R, color='green', linestyle='--', label='Poisson', alpha=0.7)
    axs[0, 0].set_title('Hamiltonian <r> (filtered)', size=TITLE_SIZE)
    axs[0, 0].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[0, 0].set_ylabel('<r>', size=LABEL_SIZE)
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot H z-statistic
    axs[0, 1].errorbar(disorder_values, H_z_mean, yerr=H_z_std,
                        label=f'L={L}', marker='o', capsize=3, color=COLORS['H'])
    axs[0, 1].axhline(y=GOE_Z, color='red', linestyle='--', label='GOE', alpha=0.7)
    axs[0, 1].axhline(y=POISSON_Z, color='green', linestyle='--', label='Poisson', alpha=0.7)
    axs[0, 1].set_title('Hamiltonian <z> (filtered)', size=TITLE_SIZE)
    axs[0, 1].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[0, 1].set_ylabel('<z>', size=LABEL_SIZE)
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot SL r-statistic
    axs[1, 0].errorbar(disorder_values, SL_r_mean, yerr=SL_r_std,
                        label=f'L={L}', marker='o', capsize=3, color=COLORS['SL'])
    axs[1, 0].axhline(y=GOE_R, color='red', linestyle='--', label='GOE', alpha=0.7)
    axs[1, 0].axhline(y=POISSON_R, color='green', linestyle='--', label='Poisson', alpha=0.7)
    axs[1, 0].set_title('Spectral Localizer <r> (filtered)', size=TITLE_SIZE)
    axs[1, 0].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[1, 0].set_ylabel('<r>', size=LABEL_SIZE)
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot SL z-statistic
    axs[1, 1].errorbar(disorder_values, SL_z_mean, yerr=SL_z_std,
                        label=f'L={L}', marker='o', capsize=3, color=COLORS['SL'])
    axs[1, 1].axhline(y=GOE_Z, color='red', linestyle='--', label='GOE', alpha=0.7)
    axs[1, 1].axhline(y=POISSON_Z, color='green', linestyle='--', label='Poisson', alpha=0.7)
    axs[1, 1].set_title('Spectral Localizer <z> (filtered)', size=TITLE_SIZE)
    axs[1, 1].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[1, 1].set_ylabel('<z>', size=LABEL_SIZE)
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    fig.suptitle(f'Filtered Spectral Statistics (|E| < E_c), L={L}', fontsize=SUPTITLE_SIZE)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axs


def plot_rz_statistics_unfiltered(data, save_path=None):
    """
    Plot r and z statistics WITHOUT mobility edge filtering.

    Uses all eigenvalues (no IPR-based filtering).
    This is the fallback when COMPUTE_IPR=False.
    """
    fig, axs = plt.subplots(2, 2, figsize=FIGSIZE_2x2, constrained_layout=True)

    L = data['L']
    disorder_values = data['disorder_values']
    n_disorder = len(disorder_values)
    n_realizations = data['H_eigval'].shape[1]

    # Arrays to store results
    H_r_vals = np.zeros((n_disorder, n_realizations))
    H_z_vals = np.zeros((n_disorder, n_realizations))
    SL_r_vals = np.zeros((n_disorder, n_realizations))
    SL_z_vals = np.zeros((n_disorder, n_realizations))

    for d_idx in range(n_disorder):
        for r_idx in range(n_realizations):
            H_eigval = data['H_eigval'][d_idx, r_idx]
            SL_eigval = data['SL_eigval'][d_idx, r_idx]

            # Calculate r and z for ALL eigenvalues (no filtering)
            H_r_vals[d_idx, r_idx] = calculate_r(H_eigval)
            H_z_vals[d_idx, r_idx] = calculate_z(H_eigval)
            SL_r_vals[d_idx, r_idx] = calculate_r(SL_eigval)
            SL_z_vals[d_idx, r_idx] = calculate_z(SL_eigval)

    # Compute means and standard errors
    H_r_mean = np.nanmean(H_r_vals, axis=1)
    H_r_std = np.nanstd(H_r_vals, axis=1) / np.sqrt(n_realizations)
    H_z_mean = np.nanmean(H_z_vals, axis=1)
    H_z_std = np.nanstd(H_z_vals, axis=1) / np.sqrt(n_realizations)

    SL_r_mean = np.nanmean(SL_r_vals, axis=1)
    SL_r_std = np.nanstd(SL_r_vals, axis=1) / np.sqrt(n_realizations)
    SL_z_mean = np.nanmean(SL_z_vals, axis=1)
    SL_z_std = np.nanstd(SL_z_vals, axis=1) / np.sqrt(n_realizations)

    # Plot H r-statistic
    axs[0, 0].errorbar(disorder_values, H_r_mean, yerr=H_r_std,
                        label=f'L={L}', marker='o', capsize=3, color=COLORS['H'])
    axs[0, 0].axhline(y=GOE_R, color='red', linestyle='--', label='GOE', alpha=0.7)
    axs[0, 0].axhline(y=POISSON_R, color='green', linestyle='--', label='Poisson', alpha=0.7)
    axs[0, 0].set_title('Hamiltonian <r>', size=TITLE_SIZE)
    axs[0, 0].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[0, 0].set_ylabel('<r>', size=LABEL_SIZE)
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot H z-statistic
    axs[0, 1].errorbar(disorder_values, H_z_mean, yerr=H_z_std,
                        label=f'L={L}', marker='o', capsize=3, color=COLORS['H'])
    axs[0, 1].axhline(y=GOE_Z, color='red', linestyle='--', label='GOE', alpha=0.7)
    axs[0, 1].axhline(y=POISSON_Z, color='green', linestyle='--', label='Poisson', alpha=0.7)
    axs[0, 1].set_title('Hamiltonian <z>', size=TITLE_SIZE)
    axs[0, 1].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[0, 1].set_ylabel('<z>', size=LABEL_SIZE)
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot SL r-statistic
    axs[1, 0].errorbar(disorder_values, SL_r_mean, yerr=SL_r_std,
                        label=f'L={L}', marker='o', capsize=3, color=COLORS['SL'])
    axs[1, 0].axhline(y=GOE_R, color='red', linestyle='--', label='GOE', alpha=0.7)
    axs[1, 0].axhline(y=POISSON_R, color='green', linestyle='--', label='Poisson', alpha=0.7)
    axs[1, 0].set_title('Spectral Localizer <r>', size=TITLE_SIZE)
    axs[1, 0].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[1, 0].set_ylabel('<r>', size=LABEL_SIZE)
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot SL z-statistic
    axs[1, 1].errorbar(disorder_values, SL_z_mean, yerr=SL_z_std,
                        label=f'L={L}', marker='o', capsize=3, color=COLORS['SL'])
    axs[1, 1].axhline(y=GOE_Z, color='red', linestyle='--', label='GOE', alpha=0.7)
    axs[1, 1].axhline(y=POISSON_Z, color='green', linestyle='--', label='Poisson', alpha=0.7)
    axs[1, 1].set_title('Spectral Localizer <z>', size=TITLE_SIZE)
    axs[1, 1].set_xlabel('Disorder Strength W', size=LABEL_SIZE)
    axs[1, 1].set_ylabel('<z>', size=LABEL_SIZE)
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    fig.suptitle(f'Spectral Statistics (all eigenvalues), L={L}', fontsize=SUPTITLE_SIZE)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axs


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='3D Anderson Model Mobility Edge Analysis')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Directory containing data files')
    parser.add_argument('--figure-dir', type=str, default=FIGURE_DIR,
                        help='Directory to save figures')
    parser.add_argument('--L', type=int, nargs='+', default=None,
                        help='System sizes to analyze (default: all available)')
    parser.add_argument('--disorder-indices', type=int, nargs='+', default=None,
                        help='Disorder indices for DOS/IPR plots')
    parser.add_argument('--ipr-threshold', type=float, default=None,
                        help='IPR threshold for mobility edge detection')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')

    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    figure_dir = os.path.join(script_dir, args.figure_dir) if not os.path.isabs(args.figure_dir) else args.figure_dir

    os.makedirs(figure_dir, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Figure directory: {figure_dir}")

    # Find and load data
    data_files = find_data_files(data_dir)

    if not data_files:
        print("No data files found!")
        return

    print(f"Found data for L = {list(data_files.keys())}")

    # Determine which L values to analyze
    L_values = args.L if args.L else list(data_files.keys())

    # Load all data
    all_data = {}
    for L in L_values:
        if L in data_files:
            print(f"Loading data for L={L}...")
            all_data[L] = load_data(data_files, L)

            # Check for eigenvector availability
            if not all_data[L].get('has_eigvec', False):
                print(f"  WARNING: No eigenvector files found for L={L}")
                print(f"           IPR will need to be computed from eigenvectors.")
                print(f"           Looking for: *_H_eigvec.dat, *_spectral_localiser_eigvec.dat")

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate plots for each L
    for L, data in all_data.items():
        print(f"\nGenerating plots for L={L}...")

        # Determine disorder indices to plot (priority: command-line > config > auto)
        if args.disorder_indices:
            disorder_indices = args.disorder_indices
        elif DISORDER_INDICES is not None:
            disorder_indices = DISORDER_INDICES
        else:
            # Default: low, mid, high disorder
            n_disorder = len(data['disorder_values'])
            disorder_indices = [0, n_disorder // 2, n_disorder - 1]

        # Figure 1: DOS and IPR summary for each disorder value
        for d_idx in disorder_indices:
            save_path = os.path.join(figure_dir, f'3dAnderson_L{L}_W{data["disorder_values"][d_idx]:.1f}_dos_ipr_{timestamp}.png')
            plot_dos_ipr_summary(data, d_idx, save_path=save_path)

        # Figure 2: Energy-resolved IPR (only if COMPUTE_IPR=True)
        if COMPUTE_IPR:
            save_path = os.path.join(figure_dir, f'3dAnderson_L{L}_energy_resolved_ipr_{timestamp}.png')
            plot_energy_resolved_ipr(data, disorder_indices, save_path=save_path)

        # Figure 4: r/z statistics (filtered if COMPUTE_IPR=True, unfiltered otherwise)
        if COMPUTE_IPR:
            save_path = os.path.join(figure_dir, f'3dAnderson_L{L}_filtered_rz_{timestamp}.png')
            plot_filtered_rz_statistics(data, ipr_threshold=args.ipr_threshold, save_path=save_path)
        else:
            save_path = os.path.join(figure_dir, f'3dAnderson_L{L}_rz_{timestamp}.png')
            plot_rz_statistics_unfiltered(data, save_path=save_path)

    # Figure 3: Mobility edge trajectory (comparing all L values, only if COMPUTE_IPR=True)
    if COMPUTE_IPR and len(all_data) > 0:
        save_path = os.path.join(figure_dir, f'3dAnderson_mobility_edge_trajectory_{timestamp}.png')
        plot_mobility_edge_trajectory(all_data, ipr_threshold=args.ipr_threshold, save_path=save_path)

    print(f"\nAll figures saved to {figure_dir}")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()

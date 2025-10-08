import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import datetime
import os

datalocation = '../data/'
figure_destination = '../figures/'


file_pattern = os.path.join(datalocation, '1dSSH_L*_rho30.0_kappa0.1_disorder0.1-5.0_numEigs600_realizations100_results.npz')
file_list = sorted(glob.glob(file_pattern))

if not file_list:
    print(f"Error: No data files found matching the pattern in '{datalocation}'.")
    print(f"Pattern used: {file_pattern}")
    exit()

print(f"Found {len(file_list)} files to analyse.")

fig, axs = plt.subplots(2, 2, figsize=(18, 18), constrained_layout=True)

for filepath in file_list:
    match = re.search(r'_L(\d+)(?:-\d+)?_', filepath)
    if not match:
        print(f"Warning: Could not extract L-value from filename: {os.path.basename(filepath)}. Skipping.")
        continue
    L = int(match.group(1))

    try:
        data = np.load(filepath)
        disorder_values = data['disorder_values']
        hr_results = data['hr_results']
        hz_results = data['hz_results']
        slr_results = data['slr_results']
        slz_results = data['slz_results']

        if hr_results.ndim == 3 and hr_results.shape[0] == 1:
            hr_results = np.squeeze(hr_results, axis=0)
            hz_results = np.squeeze(hz_results, axis=0)
            slr_results = np.squeeze(slr_results, axis=0)
            slz_results = np.squeeze(slz_results, axis=0)

    except Exception as e:
        print(f"Error loading data from {os.path.basename(filepath)}: {e}. Skipping.")
        continue

    num_realizations = hr_results.shape[1]
    sqrt_num_realizations = np.sqrt(num_realizations)

    hr_means = [hr_results[j, :].mean() for j in range(len(disorder_values))]
    hr_stds = [hr_results[j, :].std() / sqrt_num_realizations for j in range(len(disorder_values))]

    hz_means = [hz_results[j, :].mean() for j in range(len(disorder_values))]
    hz_stds = [hz_results[j, :].std() / sqrt_num_realizations for j in range(len(disorder_values))]

    slr_means = [slr_results[j, :].mean() for j in range(len(disorder_values))]
    slr_stds = [slr_results[j, :].std() / sqrt_num_realizations for j in range(len(disorder_values))]

    slz_means = [slz_results[j, :].mean() for j in range(len(disorder_values))]
    slz_stds = [slz_results[j, :].std() / sqrt_num_realizations for j in range(len(disorder_values))]

    axs[0, 0].errorbar(disorder_values, hr_means, yerr=hr_stds, label=f'L={L}', marker='o', capsize=3, linestyle='-')
    axs[0, 1].errorbar(disorder_values, hz_means, yerr=hz_stds, label=f'L={L}', marker='o', capsize=3, linestyle='-')
    axs[1, 0].errorbar(disorder_values, slr_means, yerr=slr_stds, label=f'L={L}', marker='o', capsize=3, linestyle='-')
    axs[1, 1].errorbar(disorder_values, slz_means, yerr=slz_stds, label=f'L={L}', marker='o', capsize=3, linestyle='-')

# Add horizontal lines for r plots (left column)
for ax in [axs[0, 0], axs[1, 0]]:
    ax.axhline(y=0.5295, color='red', linestyle='--', label='GOE', alpha=0.7)
    ax.axhline(y=0.386, color='green', linestyle='--', label='Poisson', alpha=0.7)
    ax.set_xlabel('Disorder Strength')
    ax.set_ylabel('r')
    ax.legend()
    ax.grid(True)

# Add horizontal lines for z plots (right column)
for ax in [axs[0, 1], axs[1, 1]]:
    ax.axhline(y=0.5687, color='red', linestyle='--', label='GOE', alpha=0.7)
    ax.axhline(y=0.5, color='green', linestyle='--', label='Poisson', alpha=0.7)
    ax.set_xlabel('Disorder Strength')
    ax.set_ylabel('z')
    ax.legend()
    ax.grid(True)

# Set titles for each subplot
axs[0, 0].set_title('r of Hamiltonian')
axs[0, 1].set_title('z of Hamiltonian')
axs[1, 0].set_title('r of Spectral Localizer')
axs[1, 1].set_title('z of Spectral Localizer')

fig.suptitle('Analysis of Hamiltonian and Spectral Localizer Statistics', fontsize=20)


# Generate a safe filename with the current date and time
now = datetime.datetime.now()
filename = '1dSSH_analysis_' + now.strftime("%Y%m%d_%H%M%S") + '.png'
plt.savefig(os.path.join(figure_destination, filename))

print(f"\nPlot saved to: {os.path.join(figure_destination, filename)}")
plt.show()


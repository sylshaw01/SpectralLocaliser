import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
import glob
import re
import os


datalocation = '../data/'
figure_destination = '../figures/'


daterange_start_str = '20250930'
daterange_end_str = '20251130'

start_date = dt.datetime.strptime(daterange_start_str, '%Y%m%d')
end_date = dt.datetime.strptime(daterange_end_str, '%Y%m%d')

file_pattern = os.path.join(datalocation, '1dAA_L*_rho30.0_kappa0.1_disorder0.1-3.0_*_results.npz')
initial_file_list = sorted(glob.glob(file_pattern))

# --- Filter the list based on the date range ---
filtered_list = []
for filepath in initial_file_list:
    # FIXED: Updated regex to handle date-time format (YYYY-MM-DD-HHMMSS)
    match = re.search(r'_(\d{4}\d{2}\d{2})-\d{6}_results\.npz$', filepath)
    if match:
        date_str = match.group(1)  # This extracts just the date part (YYYY-MM-DD)
        
        file_date = dt.datetime.strptime(date_str, '%Y%m%d')
        print(file_date)
        # Check if the file's date is within the desired range
        if start_date <= file_date <= end_date:
            filtered_list.append(filepath)
            print(f"Added file: {os.path.basename(filepath)}")  # Debug: show which files are added

# Debug: Check if any files were found
print(f"\nTotal files found: {len(initial_file_list)}")
print(f"Files after date filtering: {len(filtered_list)}")
if len(filtered_list) == 0:
    print("WARNING: No files found in the date range!")
    print("\nFiles found before filtering:")
    for f in initial_file_list[:5]:  # Show first 5 files
        print(f"  {os.path.basename(f)}")

fig, axs = plt.subplots(2, 2, figsize=(18, 18), constrained_layout=True)

for filepath in filtered_list:
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
now = dt.datetime.now()
filename = '1dAA_analysis_' + now.strftime("%Y%m%d_%H%M%S") + '.png'
plt.savefig(os.path.join(figure_destination, filename), dpi=400)

print(f"\nPlot saved to: {os.path.join(figure_destination, filename)}")
plt.show()
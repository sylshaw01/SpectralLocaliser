import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import datetime as dt
import os

datalocation = '../data/'
figure_destination = '../figures/'


daterange_start_str = '2025-11-11-01-00-00'
daterange_end_str = '2025-11-25-23-59-59'

start_date = dt.datetime.strptime(daterange_start_str, '%Y-%m-%d-%H-%M-%S')
end_date = dt.datetime.strptime(daterange_end_str, '%Y-%m-%d-%H-%M-%S')

file_pattern = os.path.join(datalocation, '1dAnderson_L*_rho30.0_kappa0.1_disorder*_results.npz')
initial_file_list = sorted(glob.glob(file_pattern))
# --- Filter the list based on the date range ---
filtered_list = []
for filepath in initial_file_list:
    # Use a regular expression to extract the datetime (YYYYMMDD-HHMMSS)
    match = re.search(r'_(\d{8}-\d{6})_results\.npz$', filepath)
    if match:
        datetime_str = match.group(1)
        # Parse the datetime string
        file_datetime = dt.datetime.strptime(datetime_str, '%Y%m%d-%H%M%S')
        # Compare only the date part (ignoring time)
        if start_date.date() <= file_datetime.date() <= end_date.date():
            filtered_list.append(filepath)




fig, axs = plt.subplots(2, 2, figsize=(18, 18), constrained_layout=True)

data_by_L = {}

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
    
    # Initialize list for this L if not exists
    if L not in data_by_L:
        data_by_L[L] = {
            'disorder': [],
            'hr': [],
            'hz': [],
            'slr': [],
            'slz': []
        }
    
    # Append data from this file
    num_realizations = hr_results.shape[1]
    sqrt_num_realizations = np.sqrt(num_realizations)
    
    for j in range(len(disorder_values)):
        data_by_L[L]['disorder'].append(disorder_values[j])
        data_by_L[L]['hr'].append((hr_results[j, :].mean(), hr_results[j, :].std() / sqrt_num_realizations))
        data_by_L[L]['hz'].append((hz_results[j, :].mean(), hz_results[j, :].std() / sqrt_num_realizations))
        data_by_L[L]['slr'].append((slr_results[j, :].mean(), slr_results[j, :].std() / sqrt_num_realizations))
        data_by_L[L]['slz'].append((slz_results[j, :].mean(), slz_results[j, :].std() / sqrt_num_realizations))

# Now plot the accumulated data
for L in sorted(data_by_L.keys()):
    d = data_by_L[L]
    
    # Sort by disorder value to ensure connected lines
    sorted_indices = np.argsort(d['disorder'])
    disorder_sorted = np.array(d['disorder'])[sorted_indices]
    
    hr_means = np.array([d['hr'][i][0] for i in sorted_indices])
    hr_stds = np.array([d['hr'][i][1] for i in sorted_indices])
    
    hz_means = np.array([d['hz'][i][0] for i in sorted_indices])
    hz_stds = np.array([d['hz'][i][1] for i in sorted_indices])
    
    slr_means = np.array([d['slr'][i][0] for i in sorted_indices])
    slr_stds = np.array([d['slr'][i][1] for i in sorted_indices])
    
    slz_means = np.array([d['slz'][i][0] for i in sorted_indices])
    slz_stds = np.array([d['slz'][i][1] for i in sorted_indices])
    
    axs[0, 0].errorbar(disorder_sorted, hr_means, yerr=hr_stds, label=f'L={L}', marker='o', capsize=3, linestyle='-')
    axs[0, 1].errorbar(disorder_sorted, hz_means, yerr=hz_stds, label=f'L={L}', marker='o', capsize=3, linestyle='-')
    axs[1, 0].errorbar(disorder_sorted, slr_means, yerr=slr_stds, label=f'L={L}', marker='o', capsize=3, linestyle='-')
    axs[1, 1].errorbar(disorder_sorted, slz_means, yerr=slz_stds, label=f'L={L}', marker='o', capsize=3, linestyle='-')

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
filename = '1dAnderson_analysis_' + now.strftime("%Y%m%d_%H%M%S") + '.png'
plt.savefig(os.path.join(figure_destination, filename))

print(f"\nPlot saved to: {os.path.join(figure_destination, filename)}")
plt.show()


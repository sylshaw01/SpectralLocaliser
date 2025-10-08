import numpy as np
import matplotlib.pyplot as plt

datalocation = '../data/'
import sys

# load data
data = np.load(datalocation + '../data/3dAnderson_L8-8_rho60.0_kappa0.1_disorder2.0-20.0_numEigs400_realizations100_results.npz')
L_values = data['L_values']
disorder_values = data['disorder_values']
hr_results = data['hr_results']
hz_results = data['hz_results']
slr_results = data['slr_results']
slz_results = data['slz_results']

# plot data
fig, axs = plt.subplots(2, 2, figsize=(18, 18))
for i, L in enumerate(L_values):
    hr_means = [hr_results[i][j].mean() for j in range(len(disorder_values))]
    hr_stds = [hr_results[i][j].std()/np.sqrt(hr_results.shape[2]) for j in range(len(disorder_values))]
    axs[0, 0].errorbar(disorder_values, hr_means, yerr=hr_stds, label=f'L={L}', marker='o', capsize=3)

    hz_means = [hz_results[i][j].mean() for j in range(len(disorder_values))]
    hz_stds = [hz_results[i][j].std()/np.sqrt(hz_results.shape[2]) for j in range(len(disorder_values))]
    axs[0, 1].errorbar(disorder_values, hz_means, yerr=hz_stds, label=f'L={L}', marker='o', capsize=3)

    slr_means = [slr_results[i][j].mean() for j in range(len(disorder_values))]
    slr_stds = [slr_results[i][j].std()/np.sqrt(slr_results.shape[2]) for j in range(len(disorder_values))]
    axs[1, 0].errorbar(disorder_values, slr_means, yerr=slr_stds, label=f'L={L}', marker='o', capsize=3)

    slz_means = [slz_results[i][j].mean() for j in range(len(disorder_values))]
    slz_stds = [slz_results[i][j].std()/np.sqrt(slz_results.shape[2]) for j in range(len(disorder_values))]
    axs[1, 1].errorbar(disorder_values, slz_means, yerr=slz_stds, label=f'L={L}', marker='o', capsize=3)
axs[0, 0].set_title('r of Hamiltonian')
axs[0, 0].set_xlabel('Disorder Strength')
axs[0, 0].set_ylabel('r')
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 1].set_title('z of Hamiltonian')
axs[0, 1].set_xlabel('Disorder Strength')
axs[0, 1].set_ylabel('z')
axs[0, 1].legend()
axs[0, 1].grid()
axs[1, 0].set_title('r of Spectral Localizer')
axs[1, 0].set_xlabel('Disorder Strength')
axs[1, 0].set_ylabel('r')
axs[1, 0].legend()
axs[1, 0].grid()
axs[1, 1].set_title('z of Spectral Localizer')
axs[1, 1].set_xlabel('Disorder Strength')
axs[1, 1].set_ylabel('z')
axs[1, 1].legend()
axs[1, 1].grid()
plt.tight_layout()
plt.savefig('3dAnderson_analysis-12-14.png', dpi=300)
plt.show()
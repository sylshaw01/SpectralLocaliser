import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
#mport matplotlib.pyplot as plt
from multiprocessing import cpu_count
#multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Pool
import time
import datetime
import sys
sys.path.append('../src')
from SLmodels import *

def single_iteration(args):
    L, rho, kappa, disorder, num_eigenvalues, X, sparse, i = args
    m = ThreeDimensionalAnderson(L, disorder, rho, kappa)
    hr, hz = m.compute_statistics(m.H,num_eigenvalues,sparse)
    slr, slz = m.compute_statistics(m.SL,num_eigenvalues,sparse)
    if i % 100 ==0:
        print(f"            Completed {i} calculations")
    return hr, hz, slr, slz



if __name__ == "__main__":
    cpu_count = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()
    parameters_file = sys.argv[2] if len(sys.argv) > 2 else 'parameters_3dAnderson.txt'

    print("-"*50)
    print("Calculating 3D Anderson model Hamiltonian and spectral localiser statistics")
    print("-"*50)

    parameters = {}
    with open(parameters_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.split('=')
                parameters[key.strip()] = value.strip()
    L_start = int(parameters.get('L_start', 5))
    L_end = int(parameters.get('L_end', 10))
    L_resolution = int(parameters.get('L_resolution', 6))
    num_disorder_realizations = int(parameters.get('num_disorder_realisations', 100))

    rho = float(parameters.get('rho', 30.0))
    kappa = float(parameters.get('kappa', 0.1))
    disorder_start = float(parameters.get('disorder_start', 2))
    disorder_end = float(parameters.get('disorder_end', 20))
    disorder_resolution = int(parameters.get('disorder_resolution', 10))
    num_eigenvalues = int(parameters.get('num_eigenvalues', 600))

    print(f"{cpu_count} Cores found! Using...All of them!!")
    print(f"Parameters loaded from {parameters_file}")
    print(f"L from {L_start} to {L_end} with {L_resolution} steps")
    print(f"Disorder from {disorder_start} to {disorder_end} with {disorder_resolution} steps")
    print(f"{num_disorder_realizations} disorder realizations per parameter set")
    print(f"Calculating {num_eigenvalues} eigenvalues per run")
    print("-"*50, flush=True)

    np.random.seed(int(parameters.get('seed', 99)))

    L_values = np.linspace(L_start, L_end, L_resolution, dtype=int)
    disorder_values = np.linspace(disorder_start, disorder_end, disorder_resolution)
    total_calculations = len(L_values) * len(disorder_values) * num_disorder_realizations
    print(f"Total calculations to be performed: {total_calculations}")

    hr_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realizations))
    hz_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realizations))
    slr_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realizations))
    slz_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realizations))

    with Pool(processes=cpu_count, maxtasksperchild=10) as pool:
        for i, L in enumerate(L_values):
            print(f"System size L: {L}", flush=True)
            modelToGetX =  ThreeDimensionalAnderson(L,0,rho,kappa)
            X = modelToGetX.X
            sparse = True
            num_eig = num_eigenvalues
            for j, disorder in enumerate(disorder_values):
                print(f"   Disorder: {disorder}", flush=True)
                args_list  = [(L, rho, kappa, disorder, num_eig, X, sparse, i) for i in range(num_disorder_realizations)]
                results = list(pool.imap(single_iteration, args_list, chunksize=1))
                hr_values, hz_values, slr_values, slz_values = zip(*results)
                hr_results[i, j, :] = hr_values
                hz_results[i, j, :] = hz_values
                slr_results[i, j, :] = slr_values
                slz_results[i, j, :] = slz_values
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = f"../data/3dAnderson_L{L_start}-{L_end}_rho{rho}_kappa{kappa}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realizations}_results.npz"
    np.savez(filename, L_values = L_values, disorder_values = disorder_values, hr_results = hr_results, hz_results = hz_results, slr_results = slr_results, slz_results = slz_results)
    print(f"Results saved to {filename}", flush=True)







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
import sys
sys.path.append('../src')
from SLmodels import *

def single_iteration(args):
    L, rho, kappa, disorder, num_eigenvalues, X, sparse, v,w, i = args
    seed = int(rho * 10e5) + int(disorder * 10e7) + int(num_eigenvalues*10e4) + i
    np.random.seed(seed)
    m = OneDimensionalAubryAndre(L, disorder, rho, kappa,X)
    hr, hz = m.compute_statistics(m.H,num_eigenvalues,sparse,1e-7,False)
    slr, slz = m.compute_statistics(m.SL,2 * num_eigenvalues,sparse,1e-7, False)
    if i % 500 ==0:
        print(f"            Completed {i} calculations")
    return hr, hz, slr, slz, seed



if __name__ == "__main__":
    cpu_count = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()
    parameters_file = sys.argv[2] if len(sys.argv) > 2 else 'parameters_1dSSH.txt'

    print("-"*50)
    print("Calculating 1D SSH model Hamiltonian and spectral localiser statistics")
    print("-"*50)

    parameters = {}
    with open(parameters_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.split('=')
                parameters[key.strip()] = value.strip()
    L_start = int(parameters.get('L_start', 200))
    L_end = int(parameters.get('L_end', 400))
    L_resolution = int(parameters.get('L_resolution', 3))
    num_disorder_realisations = int(parameters.get('num_disorder_realisations', 100))

    rho = float(parameters.get('rho', 30.0))
    kappa = float(parameters.get('kappa', 0.1))
    v = float(parameters.get('v', 0.5))
    w = float(parameters.get('w', 1.0))
    disorder_start = float(parameters.get('disorder_start', 0.0))
    disorder_end = float(parameters.get('disorder_end', 5.0))
    disorder_resolution = int(parameters.get('disorder_resolution', 6))
    num_eigenvalues = int(parameters.get('num_eigenvalues', 600))

    print(f"{cpu_count} Cores found! Using...All of them!!")
    print(f"Parameters loaded from {parameters_file}")
    print(f"L from {L_start} to {L_end} with {L_resolution} steps")
    print(f"Disorder from {disorder_start} to {disorder_end} with {disorder_resolution} steps")
    print(f"{num_disorder_realisations} disorder realizations per parameter set")
    print(f"Calculating {num_eigenvalues} eigenvalues per run")
    print("-"*50, flush=True)



    L_values = np.linspace(L_start, L_end, L_resolution, dtype=int)
    disorder_values = np.linspace(disorder_start, disorder_end, disorder_resolution)
    total_calculations = len(L_values) * len(disorder_values) * num_disorder_realisations
    print(f"Total calculations to be performed: {total_calculations}")

    hr_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realisations))
    hz_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realisations))
    slr_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realisations))
    slz_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realisations))
    seeds = np.zeros((len(L_values), len(disorder_values), num_disorder_realisations))
    total_time = time.time()


    with Pool(processes=cpu_count, maxtasksperchild=10) as pool:
        for i, L in enumerate(L_values):
            L_val_time = time.time()
            print(f"System size L: {L}", flush=True)
            modelToGetX =  OneDimensionalAubryAndre(L,0,rho,kappa)
            X = modelToGetX.X
            sparse = True
            num_eig = num_eigenvalues
            for j, disorder in enumerate(disorder_values):
                print(f"   Disorder: {disorder}", flush=True)
                args_list  = [(L, rho, kappa, disorder, num_eig, X, sparse,v,w, i) for i in range(num_disorder_realisations)]
                results = list(pool.imap(single_iteration, args_list, chunksize=1))
                print(f"      Time for disorder {disorder}: {time.time() - L_val_time} seconds", flush=True)
                hr_values, hz_values, slr_values, slz_values, seed_values = zip(*results)
                hr_results[i, j, :] = hr_values
                hz_results[i, j, :] = hz_values
                slr_results[i, j, :] = slr_values
                slz_results[i, j, :] = slz_values
                seeds[i, j, :] = seed_values
            print(f"   Time taken for all disorder values at L={L}: {time.time() - L_val_time} seconds", flush=True)
    print(f"Total time for all calculations: {time.time() - total_time} seconds", flush=True)   
    

    import datetime

    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = f"../data/1dAA_L{L_start}-{L_end}_rho{rho}_kappa{kappa}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realisations}_{current_date}_results.npz"
    np.savez(filename, L_values = L_values, disorder_values = disorder_values, hr_results = hr_results, hz_results = hz_results, slr_results = slr_results, slz_results = slz_results, seeds = seeds)
    print(f"Results saved to {filename}", flush=True)







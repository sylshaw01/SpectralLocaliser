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
    L, rho, kappa, disorder, num_eigenvalues, X, sparse, reteval, retevec, v,w, i = args
    seed = int(rho * 10e5) + int(disorder * 10e7) + int(num_eigenvalues*10e4) + i
    np.random.seed(seed)
    m = OneDimensionalSSHBlockBasis(L, disorder, rho, kappa,v,w,X)
    hr, hz, hevals = m.compute_statistics(m.H,num_eigenvalues=num_eigenvalues,sparse=sparse,tolerance=1e-7,slepc=False, returneVals=reteval, returneVecs=retevec)
    slr, slz, slevals = m.compute_statistics(m.SL,num_eigenvalues=2 * num_eigenvalues,sparse=sparse,tolerance=1e-7,slepc= False,returneVals=reteval, returneVecs=retevec)
    if i % 500 ==0:
        print(f"            Completed {i} calculations")
    return hr, hz, slr, slz, hevals, slevals, seed



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
    reteval = True
    retevec = False

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

    hr_results = np.zeros(( len(disorder_values), num_disorder_realisations))
    hz_results = np.zeros(( len(disorder_values), num_disorder_realisations))
    hev_results = np.zeros((len(disorder_values), num_disorder_realisations, L_start))
    slr_results = np.zeros(( len(disorder_values), num_disorder_realisations))
    slz_results = np.zeros(( len(disorder_values), num_disorder_realisations))
    slev_results = np.zeros((len(disorder_values),num_disorder_realisations,2 * L_start))
    seeds = np.zeros(( len(disorder_values), num_disorder_realisations))
    total_time = time.time()



    L = L_start
    with Pool(processes=cpu_count, maxtasksperchild=10) as pool:
        modelToGetX =  OneDimensionalSSHBlockBasis(L,0,rho,kappa,v,w)
        X = modelToGetX.X
        sparse = False
        num_eig = num_eigenvalues
        for j, disorder in enumerate(disorder_values):
            print(f"   Disorder: {disorder}", flush=True)
            disorder_time = time.time()
            args_list  = [(L, rho, kappa, disorder, num_eig, X, sparse,reteval, retevec, v,w, i) for i in range(num_disorder_realisations)]
            results = list(pool.imap(single_iteration, args_list, chunksize=1))
            print(f"      Time for disorder {disorder}: {time.time() - disorder_time} seconds", flush=True)
            hr_values, hz_values, slr_values, slz_values, hev, slev, seed_values = zip(*results)
            hr_results[ j, :] = hr_values
            hz_results[ j, :] = hz_values
            slr_results[ j, :] = slr_values
            slz_results[ j, :] = slz_values
            seeds[ j, :] = seed_values
            hev_results[j,:] = hev
            slev_results[j,:] = slev
    print(f"Total time for all calculations: {time.time() - total_time} seconds", flush=True)   
    

    import datetime

    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = f"../data/1dSSH_L{L_start}-{L_end}_rho{rho}_kappa{kappa}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realisations}_{current_date}_results.npz"
    np.savez(filename, L_values = L_values, disorder_values = disorder_values, hr_results = hr_results, hz_results = hz_results, slr_results = slr_results, slz_results = slz_results, seeds = seeds, hevals_results = hev_results, slev_results=slev_results)
    print(f"Results saved to {filename}", flush=True)







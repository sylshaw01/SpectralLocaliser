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
    L, rho, kappa, disorder, num_eigenvalues, X, sparse,retevals, retevecs,  i = args

    seed = int(rho * 10e5) + int(disorder * 10e7) + int(num_eigenvalues*10e4) + i
    np.random.seed(seed)
    m = OneDimensionalAubryAndre(L, disorder, rho, kappa, X)
    hr, hz , hev = m.compute_statistics(m.H,num_eigenvalues=num_eigenvalues, sparse=sparse, tolerance=1e-7, slepc=False, returneVals=retevals,returneVecs=retevecs)
    slr, slz , slev = m.compute_statistics(m.SL,num_eigenvalues = 2 * num_eigenvalues,sparse = sparse, tolerance = 1e-7, slepc = False,  returneVals=retevals, returneVecs=retevecs)
    if i % 500 ==0:
        print(f"            Completed {i} calculations")
    return hr, hz, slr, slz, hev, slev, seed



if __name__ == "__main__":
    cpu_count = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()
    parameters_file = sys.argv[2] if len(sys.argv) > 2 else 'parameters_1dAA.txt'
    print("-"*50)
    print("Calculating 1D Aubry Andre model Hamiltonian and spectral localiser statistics")
    print("-"*50)


    parameters = {}
    with open(parameters_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.split('=')
                parameters[key.strip()] = value.strip()
    L = int(parameters.get('L', 200))
    num_disorder_realizations = int(parameters.get('num_disorder_realisations', 100))
    rho = float(parameters.get('rho', 30.0))
    kappa = float(parameters.get('kappa', 0.1))
    disorder_start = float(parameters.get('disorder_start', 0.0))
    disorder_end = float(parameters.get('disorder_end', 5.0))
    disorder_resolution = int(parameters.get('disorder_resolution', 6))
    sparse = False
    num_eigenvalues = int(parameters.get('num_eigenvalues', 600))
    get_eigvals = True
    get_eigvecs = False


    np.random.seed(int(parameters.get('seed', 99)))

    print(f"{cpu_count} Cores found! Using...All of them!!")
    print(f"Parameters loaded from {parameters_file}")
    print(f"L = {L}")
    print(f"Disorder from {disorder_start} to {disorder_end} with {disorder_resolution} steps")
    print(f"{num_disorder_realizations} disorder realizations per parameter set")
    if sparse==False:
        print(f"Calculating {L} eigenvalues per run")
    else:
        print(f"Calculating {num_eigenvalues} eigenvalues per run")
    print("-"*50, flush=True)


    disorder_values = np.linspace(disorder_start, disorder_end, disorder_resolution)
    total_calculations =  len(disorder_values) * num_disorder_realizations
    print(f"Total calculations to be performed: {total_calculations}")

    hr_results = np.zeros(( len(disorder_values), num_disorder_realizations))
    hz_results = np.zeros(( len(disorder_values), num_disorder_realizations))
    slr_results = np.zeros(( len(disorder_values), num_disorder_realizations))
    slz_results = np.zeros(( len(disorder_values), num_disorder_realizations))
    seeds = np.zeros(( len(disorder_values), num_disorder_realizations))
    total_time = time.time()

    with Pool(processes=cpu_count, maxtasksperchild=10) as pool:
        L_val_time = time.time()
        print(f"System size L: {L}", flush=True)
        # Instantiate a model just to get X
        modelToGetX =  OneDimensionalAubryAndre(L,0,rho,kappa)
        X = modelToGetX.X
        SLsample = modelToGetX.SL
        if get_eigvals:
            heval_results = np.zeros((len(disorder_values),num_disorder_realizations,L))
            sleval_results = np.zeros((len(disorder_values),num_disorder_realizations,SLsample.shape[0]))
        num_eig = num_eigenvalues
        for j, disorder in enumerate(disorder_values):
            print(f"   Disorder: {disorder}", flush=True)
            args_list  = [(L, rho, kappa, disorder, num_eig, X, sparse, get_eigvals, get_eigvecs, i) for i in range(num_disorder_realizations)]
            disorder_start_time = time.time()
            results = list(pool.imap(single_iteration, args_list, chunksize=1))
            print(f"      Time taken for disorder {disorder}: {time.time() - disorder_start_time:.2f} seconds", flush=True)
            hr_values, hz_values, slr_values, slz_values,hevalues, slevalues ,seed_values  = zip(*results)
            hr_results[ j, :] = hr_values
            hz_results[ j, :] = hz_values
            slr_results[ j, :] = slr_values
            slz_results[ j, :] = slz_values
            seeds[ j, :] = seed_values
            if get_eigvals:
                heval_results[j,:] = hevalues[0]
                sleval_results[j,:] = slevalues[0]
    print(f"Total time taken for all calculations: {time.time() - total_time:.2f} seconds", flush=True)
    
    import datetime

    current_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    filename = f"../data/1dAA_L{L}_rho{rho}_kappa{kappa}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realizations}_{current_date}_results.npz"
    np.savez(filename, L_values = L, disorder_values = disorder_values, hr_results = hr_results, hz_results = hz_results, slr_results = slr_results, slz_results = slz_results, seeds = seeds, heval_results = heval_results, sleval_results = sleval_results)
    print(f"Results saved to {filename}", flush=True)







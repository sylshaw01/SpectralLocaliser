
# Set environment variables to make sure pool behaves! otherwise I think it ends up locking itself by assigning too many threads
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
    L, rho, kappa, disorder, num_eigenvalues, X, sparse, reteval, retevec, v,w, i = args
    seed = int(rho * 10e5) + int(disorder * 10e7) + int(num_eigenvalues*10e4) + i
    np.random.seed(seed)
    m = OneDimensionalSSHBlockBasis(L, disorder, rho, kappa,v,w,X)
    #hr, hz, hevals = m.compute_statistics(m.H,num_eigenvalues=num_eigenvalues,sparse=sparse,tolerance=1e-7,slepc=False, returneVals=reteval, returneVecs=retevec)
    #slr, slz, slevals = m.compute_statistics(m.SL,num_eigenvalues= num_eigenvalues,sparse=sparse,tolerance=1e-7,slepc= False,returneVals=reteval, returneVecs=retevec)
    m.find_eigenvalues(m.H, num_eigenvalues=num_eigenvalues, sparse=sparse)
    hevals = m.eigvals_H
    m.find_eigenvalues(m.SL, num_eigenvalues=num_eigenvalues, sparse=sparse)
    slevals = m.eigvals_SL
    slIPR = m.compute_IPR(m.eigvecs_SL)
    heIPR = m.compute_IPR(m.eigvecs_H)
    if i % 500 ==0:
        print(f"            Completed {i} calculations")
    return  hevals, slevals,heIPR, slIPR,  seed



if __name__ == "__main__":
    # Take cpu count and parameters file from command line arguments
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

    # Import parameters from file
    L = int(parameters.get('L', 200))
    num_disorder_realisations = int(parameters.get('num_disorder_realisations', 100))

    rho = float(parameters.get('rho', 30.0))
    kappa = float(parameters.get('kappa', 0.1))
    v = float(parameters.get('v', 1.0))
    w = float(parameters.get('w', 1.5))
    disorder_start = float(parameters.get('disorder_start', 0.0))
    disorder_end = float(parameters.get('disorder_end', 5.0))
    disorder_resolution = int(parameters.get('disorder_resolution', 6))
    # Number of eigenvalues to compute if we are using sparse methods
    num_eigenvalues = int(parameters.get('num_eigenvalues', 600))

    # Boolean flags to get eigenvalues and eigenvectors
    reteval = True
    retevec = False

    print(f"{cpu_count} Cores found! Using...All of them!!")
    print(f"Parameters loaded from {parameters_file}")
    print(f"Disorder from {disorder_start} to {disorder_end} with {disorder_resolution} steps")
    print(f"{num_disorder_realisations} disorder realizations per parameter set")
    print(f"Calculating {num_eigenvalues} eigenvalues per run")
    print("-"*50, flush=True)

    disorder_values = np.linspace(disorder_start, disorder_end, disorder_resolution)

    #hr_results = np.zeros(( len(disorder_values), num_disorder_realisations))
    #hz_results = np.zeros(( len(disorder_values), num_disorder_realisations))
    hev_results = np.zeros((len(disorder_values), num_disorder_realisations,L))
    #slr_results = np.zeros(( len(disorder_values), num_disorder_realisations))
    #slz_results = np.zeros(( len(disorder_values), num_disorder_realisations))
    slev_results = np.zeros((len(disorder_values),num_disorder_realisations, L))
    hipr_results = np.zeros((len(disorder_values),num_disorder_realisations, L))
    slipr_results = np.zeros((len(disorder_values),num_disorder_realisations, L))
    seeds = np.zeros(( len(disorder_values), num_disorder_realisations))
    total_time = time.time()



    with Pool(processes=cpu_count, maxtasksperchild=10) as pool:
        modelToGetX =  OneDimensionalSSHBlockBasis(L,0,rho,kappa,v,w)
        X = modelToGetX.X
        sparse = False
        num_eig = num_eigenvalues
        for j, disorder in enumerate(disorder_values):
            print(f"   Disorder: {disorder}", flush=True)
            disorder_time = time.time()
            kappa = (3 + disorder * 0.5)/ rho # Rough value which is appropriate for kappa, for the SSH model
            args_list  = [(L, rho, kappa, disorder, num_eig, X, sparse,reteval, retevec, v,w, i) for i in range(num_disorder_realisations)]
            results = list(pool.imap(single_iteration, args_list, chunksize=1))
            print(f"      Time for disorder {disorder}: {time.time() - disorder_time} seconds", flush=True)
            hev, slev, hipr, slipr, seed_values = zip(*results)
            # hr_results[ j, :] = hr_values
            # hz_results[ j, :] = hz_values
            # slr_results[ j, :] = slr_values
            # slz_results[ j, :] = slz_values
            seeds[ j, :] = seed_values
            hev_results[j,:, :] = np.array(hev)
            slev_results[j,:, :] = np.array(slev)
            hipr_results[j,:, :] = np.array(hipr)
            slipr_results[j,:, :] = np.array(slipr)

    elapsed_time = time.time() - total_time
    print(f"Total time for all calculations: {elapsed_time:.2f} seconds, or {(elapsed_time)/60:.2f} minutes, {(elapsed_time)/3600:.2f} hours", flush=True)   
    


    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = f"../data/1dSSH_L{L}_w_{w}_rho{rho}_kappa{kappa}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realisations}_{current_date}_results.npz"
    np.savez(filename, disorder_values = disorder_values, seeds = seeds, hevals_results = hev_results, slev_results=slev_results, hipr_results=hipr_results, slipr_results=slipr_results)   
    print(f"Results saved to {filename}", flush=True)







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
    L, rho,  disorder, num_eigenvalues, X, sparse,retevals, retevecs,theta, beta,  i = args
    kappa = (2 + disorder * 0.5)/rho # Rough value which is appropriate for kappa, for the AA model
    seed = int(rho * 10e5) + int(disorder * 10e7) + int(num_eigenvalues*10e4) + i
    np.random.seed(seed)
    m = OneDimensionalAubryAndre(L, disorder, rho, kappa, X, beta, theta)
    #hr, hz , hev = m.compute_statistics(m.H,num_eigenvalues=num_eigenvalues, sparse=sparse, tolerance=1e-7, slepc=False, returneVals=retevals,returneVecs=retevecs)
    #slr, slz , slev = m.compute_statistics(m.SL,num_eigenvalues = 2 * num_eigenvalues,sparse = sparse, tolerance = 1e-7, slepc = False,  returneVals=retevals, returneVecs=retevecs)
    m.find_eigenvalues(m.H, num_eigenvalues=num_eigenvalues, sparse=sparse)
    hev = m.eigvals_H
    m.find_eigenvalues(m.SL, num_eigenvalues=2 * num_eigenvalues, sparse=sparse)
    slev = m.eigvals_SL
    hipr = m.compute_IPR(m.eigvecs_H)
    slipr = m.compute_IPR(m.eigvecs_SL)
    if i % 500 ==0:
        print(f"            Completed {i} calculations")
    return  hev, slev,hipr, slipr,  seed



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
    theta_start = float(parameters.get('theta_start', 0.0))
    theta_end = float(parameters.get('theta_end', 2.0 * np.pi))
    theta_resolution = int(parameters.get('theta_resolution', 6))
    beta = float(parameters.get('beta', (5**0.5 -1)/2))
    beta_numerator = float(parameters.get('beta_numerator', 1))
    beta_denominator = float(parameters.get('beta_denominator', 1))
    disorder_resolution = int(parameters.get('disorder_resolution', 6))
    sparse = False
    num_eigenvalues = int(parameters.get('num_eigenvalues', 600))
    get_eigvals = True
    get_eigvecs = False

    if beta < 1.0:
        beta = (5**0.5 -1)/2  # Set to golden ratio by default
    else:
        beta = beta_numerator / beta_denominator


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
    theta_values = np.linspace(theta_start, theta_end, theta_resolution)
    total_calculations =  len(disorder_values) * num_disorder_realizations
    print(f"Total calculations to be performed: {total_calculations}")

    # hr_results = np.zeros(( len(disorder_values), num_disorder_realizations))
    # hz_results = np.zeros(( len(disorder_values), num_disorder_realizations))
    # slr_results = np.zeros(( len(disorder_values), num_disorder_realizations))
    # slz_results = np.zeros(( len(disorder_values), num_disorder_realizations))

    heval_results = np.zeros((len(theta_values),len(disorder_values),L))
    sleval_results = np.zeros((len(theta_values),len(disorder_values),L*2))
    hipr_results = np.zeros((len(theta_values),len(disorder_values),L))
    slipr_results = np.zeros((len(theta_values),len(disorder_values),L*2))
    seeds = np.zeros(( len(theta_values), len(disorder_values)))
    total_time = time.time()
    
    with Pool(processes=cpu_count, maxtasksperchild=10) as pool:
        L_val_time = time.time()
        print(f"System size L: {L}", flush=True)
        # Instantiate a model just to get X
        modelToGetX =  OneDimensionalAubryAndre(L,0,rho,kappa)
        X = modelToGetX.X
        num_eig = num_eigenvalues
        for j, theta in enumerate(theta_values):
            print(f"   Theta: {theta}", flush=True)
            #kappa =  (2 + disorder * 0.5)/rho # Rough value which is appropriate for kappa, for the AA model
            args_list  = [(L, rho, disorder, num_eig, X, sparse, get_eigvals, get_eigvecs,theta,beta, i) for i , disorder in enumerate(disorder_values)]
            disorder_start_time = time.time()
            results = list(pool.imap(single_iteration, args_list, chunksize=1))
            print(f"      Time taken for theta {theta}: {time.time() - disorder_start_time:.2f} seconds", flush=True)
            hevalues, slevalues ,hipr, slipr, seed_values  = zip(*results)
            # hr_results[ j, :] = hr_values
            # hz_results[ j, :] = hz_values
            # slr_results[ j, :] = slr_values
            # slz_results[ j, :] = slz_values
            seeds[ j, :] = seed_values
            heval_results[j,:, :] = np.array(hevalues)
            sleval_results[j,:, :] = np.array(slevalues)
            hipr_results[j,:, :] = np.array(hipr)
            slipr_results[j,:, :] = np.array(slipr)

    print(f"Total time taken for all calculations: {time.time() - total_time:.2f} seconds", flush=True)
    

    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    filename = f"../data/1dAA_L{L}_rho{rho}_kappa{kappa:.2f}_beta{beta:.6f}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realizations}_{current_date}_results.npz"
    np.savez(filename, L_values = L, disorder_values = disorder_values,  seeds = seeds, heval_results = heval_results, sleval_results = sleval_results, hipr_results=hipr_results, slipr_results=slipr_results)
    print(f"Results saved to {filename}", flush=True)







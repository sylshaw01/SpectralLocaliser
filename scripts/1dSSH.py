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
    m = OneDimensionalSSHBlockBasis(L, disorder, rho, kappa,v,w)
    hr, hz = m.compute_statistics(m.H,num_eigenvalues,sparse)
    slr, slz = m.compute_statistics(m.SL,num_eigenvalues,sparse)
    if i % 500 ==0:
        print(f"            Completed {i} calculations")
    return hr, hz, slr, slz



if __name__ == "__main__":
    cpu_count = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()
    parameters_file = sys.argv[2] if len(sys.argv) > 2 else 'parameters_1dSSH.txt'

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
    num_eigenvalues = float(parameters.get('num_eigenvalues', 0.2))

    L_values = np.linspace(L_start, L_end, L_resolution, dtype=int)
    disorder_values = np.linspace(disorder_start, disorder_end, disorder_resolution)
    total_calculations = len(L_values) * len(disorder_values) * num_disorder_realisations
    print(f"Total calculations to be performed: {total_calculations}")

    hr_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realisations))
    hz_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realisations))
    slr_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realisations))
    slz_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realisations))

    with Pool(processes=cpu_count, maxtasksperchild=10) as pool:
        for i, L in enumerate(L_values):
            print(f"System size L: {L}", flush=True)
            modelToGetX =  OneDimensionalSSHBlockBasis(L,0,rho,kappa,v,w)
            X = modelToGetX.X
            sparse = True
            num_eig = 200
            if L <= 600:
                sparse = False
                num_eig = None
            for j, disorder in enumerate(disorder_values):
                print(f"   Disorder: {disorder}", flush=True)
                args_list  = [(L, rho, kappa, disorder, num_eig, X, sparse,v,w, i) for i in range(num_disorder_realisations)]
                results = list(pool.imap(single_iteration, args_list, chunksize=1))
                hr_values, hz_values, slr_values, slz_values = zip(*results)
                hr_results[i, j, :] = hr_values
                hz_results[i, j, :] = hz_values
                slr_results[i, j, :] = slr_values
                slz_results[i, j, :] = slz_values
    
    filename = f"../data/1dSSH_L{L_start}-{L_end}_rho{rho}_kappa{kappa}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realisations}_results.npz"
    np.savez(filename, L_values = L_values, disorder_values = disorder_values, hr_results = hr_results, hz_results = hz_results, slr_results = slr_results, slz_results = slz_results)
    print(f"Results saved to {filename}", flush=True)








# Set environment variables to make sure pool behaves! otherwise I think it ends up locking itself by assigning too many threads
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Normal imports
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool
import time
import datetime
import sys
import hashlib
# Import SLmodels from src
sys.path.append('../src')
from SLmodels import *

def single_iteration(args):
    # Unpack arguments
    L, rho, kappa, disorder, num_eigval, X, sparse, return_evec, return_eval, E_offset, i = args
    # Generate unique seed for reproducibility, using hashlib to avoid collisions
    time_of_day = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    seed_str = f"{rho}_{disorder}_{num_eigval}_{i}_{kappa:.5f}_{L}_{time_of_day}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    m = OneDimensionalAnderson(L, disorder, rho, kappa, X, energy_offset = E_offset)
    m.find_eigval(m.H, sparse=False)
    H_eigval = m.H_eigval
    H_eigvec = m.H_eigvec
    m.find_eigval(m.spectral_localiser, sparse=False)
    spectral_localiser_eigval = m.spectral_localiser_eigval
    spectral_localiser_eigvec = m.spectral_localiser_eigvec
    spectral_localiser_IPR = m.compute_IPR(m.spectral_localiser_eigvec)
    H_IPR = m.compute_IPR(m.H_eigvec)

    if return_evec:
        return H_eigval, spectral_localiser_eigval, H_eigvec, spectral_localiser_eigvec, H_IPR, spectral_localiser_IPR, seed
    return H_eigval, spectral_localiser_eigval, H_IPR, spectral_localiser_IPR, seed



if __name__ == "__main__":
    # Take cpu count and parameters file from command line arguments
    cpu_count = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()
    parameters_file = sys.argv[2] if len(sys.argv) > 2 else 'parameters_1dAndersonSULIS.txt'

    print("-"*50)
    print("Calculating 1D Anderson model Hamiltonian and spectral localiser statistics")
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
    file_kappa = float(parameters.get('kappa', 0.1))
    disorder_start = float(parameters.get('disorder_start', 0.0))
    disorder_end = float(parameters.get('disorder_end', 5.0))
    disorder_resolution = int(parameters.get('disorder_resolution', 6))
    # Number of eigenvalues to compute if we are using sparse methods
    num_eigenvalues = int(parameters.get('num_eigenvalues', 600))
    E_offset = float(parameters.get('E_offset', 0.0))


    disorder_values = np.linspace(disorder_start, disorder_end, disorder_resolution)

    # Boolean flags to get eigenvalues and eigenvectors
    reteval = True
    retevec = False

    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if file_kappa < 0:
        base_name = f"../data/1dAnderson_L{L}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realisations}_E_OFFSET{E_offset}_{current_date}"
    else:
        base_name = f"../data/1dAnderson_L{L}_disorder{disorder_start}-{disorder_end}_kappa{file_kappa}_numEigs{num_eigenvalues}_realizations{num_disorder_realisations}_E_OFFSET{E_offset}_{current_date}"

    # For 1D Anderson: H has size L, spectral_localiser has size 2L
    shape_4d_H = (len(disorder_values), num_disorder_realisations, L, L)
    shape_4d_SL = (len(disorder_values), num_disorder_realisations, 2*L, 2*L)
    shape_3d_H = (len(disorder_values), num_disorder_realisations, L)
    shape_3d_SL = (len(disorder_values), num_disorder_realisations, 2*L)
    shape_2d = (len(disorder_values), num_disorder_realisations)



    print(f"{cpu_count} Cores found! Using...All of them!!")
    print(f"Parameters loaded from {parameters_file}")
    print(f"L = {L}")
    print(f"Disorder from {disorder_start} to {disorder_end} with {disorder_resolution} steps")
    print(f"{num_disorder_realisations} disorder realizations per parameter set")
    print(f"Calculating all {L} (H) and {2*L} (SL) eigenvalues per run (dense solve)")
    print(f"Energy offset set to {E_offset}")
    print("-"*50, flush=True)


    # NOTE - Using dense solve, so we get all eigenvalues
    H_eigval_results = np.memmap(f"{base_name}_H_eigval.dat", dtype='float64', mode='w+', shape=shape_3d_H)
    spectral_localiser_eigval_results = np.memmap(f"{base_name}_spectral_localiser_eigval.dat", dtype='float64', mode='w+', shape=shape_3d_SL)
    H_IPR_results = np.memmap(f"{base_name}_H_IPR.dat", dtype='float64', mode='w+', shape=shape_3d_H)
    spectral_localiser_IPR_results = np.memmap(f"{base_name}_spectral_localiser_IPR.dat", dtype='float64', mode='w+', shape=shape_3d_SL)
    if retevec:
        H_eigvec_results = np.memmap(f"{base_name}_H_eigvec.dat", dtype='complex128', mode='w+', shape=shape_4d_H)
        spectral_localiser_eigvec_results = np.memmap(f"{base_name}_spectral_localiser_eigvec.dat", dtype='complex128', mode='w+', shape=shape_4d_SL)

    seeds = np.memmap(f"{base_name}_seeds.dat", dtype='int64', mode='w+', shape=shape_2d)
    total_time = time.time()


    with open(f"{base_name}_parameters.txt", 'w') as f:
        for key, value in parameters.items():
            f.write(f"{key} = {value}\n")
        f.write(f"Created at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")




    with Pool(processes=cpu_count, maxtasksperchild=10) as pool:
        modelToGetX = OneDimensionalAnderson(L, 0, rho, 1.0)
        X = modelToGetX.X
        sparse = False  # Dense solve to get all eigenvalues
        num_eigval = L  # Get all eigenvalues
        for j, disorder in enumerate(disorder_values):
            print(f"   Disorder: {disorder}", flush=True)
            disorder_time = time.time()
            # Find largest eigenvalue to set kappa appropriately
            modelforkappa = OneDimensionalAnderson(L, disorder, rho=L//2, kappa=1.0)
            largest_eigenvalue = eigsh(modelforkappa.H, k=1, which='LM', return_eigenvectors=False)[0]
            if file_kappa < 0:
                kappa = abs(largest_eigenvalue) / rho
            else:
                kappa = file_kappa  # Use fixed kappa from parameters file
            print(f"      Setting kappa to {kappa:.12f} based on largest eigenvalue {largest_eigenvalue:.12f}", flush=True)
            args_list = [(L, rho, kappa, disorder, num_eigval, X, sparse, retevec, reteval, E_offset, i) for i in range(num_disorder_realisations)]
            results = list(pool.imap(single_iteration, args_list, chunksize=1))
            print(f"      Time for disorder {disorder}: {time.time() - disorder_time} seconds", flush=True)
            if retevec:
                H_eigval, spectral_localiser_eigval, H_eigvec, spectral_localiser_eigvec, H_IPR, spectral_localiser_IPR, seed_values = zip(*results)
                seeds[j, :] = seed_values
                H_eigval_results[j, :, :] = np.array(H_eigval)
                spectral_localiser_eigval_results[j, :, :] = np.array(spectral_localiser_eigval)
                H_eigvec_results[j, :, :, :] = np.array(H_eigvec)
                spectral_localiser_eigvec_results[j, :, :, :] = np.array(spectral_localiser_eigvec)
                H_IPR_results[j, :, :] = np.array(H_IPR)
                spectral_localiser_IPR_results[j, :, :] = np.array(spectral_localiser_IPR)

                seeds.flush()
                H_eigval_results.flush()
                spectral_localiser_eigval_results.flush()
                H_eigvec_results.flush()
                spectral_localiser_eigvec_results.flush()
                H_IPR_results.flush()
                spectral_localiser_IPR_results.flush()
                continue
            else:
                H_eigval, spectral_localiser_eigval, H_IPR, spectral_localiser_IPR, seed_values = zip(*results)
                seeds[j, :] = seed_values
                H_eigval_results[j, :, :] = np.array(H_eigval)
                spectral_localiser_eigval_results[j, :, :] = np.array(spectral_localiser_eigval)
                H_IPR_results[j, :, :] = np.array(H_IPR)
                spectral_localiser_IPR_results[j, :, :] = np.array(spectral_localiser_IPR)

                seeds.flush()
                H_eigval_results.flush()
                spectral_localiser_eigval_results.flush()
                H_IPR_results.flush()
                spectral_localiser_IPR_results.flush()

    elapsed_time = time.time() - total_time
    print(f"Total time for all calculations: {elapsed_time:.2f} seconds, or {(elapsed_time)/60:.2f} minutes, {(elapsed_time)/3600:.2f} hours", flush=True)
    print("-"*50, flush=True)

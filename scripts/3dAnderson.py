
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
    L, rho, kappa, disorder, num_eigval, X, sparse,return_evec, return_eval, i = args
    # Generate unique seed for reproducibility, using hashlib to avoid collisions
    seed_str = f"{rho}_{disorder}_{num_eigval}_{i}_{kappa:.5f}_{L}"
    #seed = int(rho * 10e5) + int(disorder * 10e7) + int(num_eigval*10e4) + i OLD SEED GENERATION METHOD
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(),16) % (2**32)
    np.random.seed(seed)
    m = ThreeDimensionalAnderson(L, disorder, rho, kappa,X)
    m.find_eigval(m.H, num_eigval=num_eigval, sparse=sparse)
    H_eigval = m.H_eigval
    H_eigvec = m.H_eigvec
    m.find_eigval(m.spectral_localiser, num_eigval=num_eigval, sparse=sparse)
    spectral_localiser_eigval = m.spectral_localiser_eigval
    spectral_localiser_eigvec = m.spectral_localiser_eigvec
    spectral_localiser_IPR = m.compute_IPR(m.spectral_localiser_eigvec)
    H_IPR = m.compute_IPR(m.H_eigvec)
    if return_evec:
        return  H_eigval, spectral_localiser_eigval, H_eigvec, spectral_localiser_eigvec, seed
    return  H_eigval, spectral_localiser_eigval,H_IPR, spectral_localiser_IPR,  seed



if __name__ == "__main__":
    # Take cpu count and parameters file from command line arguments
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

    # Import parameters from file
    L = int(parameters.get('L', 200))
    num_disorder_realisations = int(parameters.get('num_disorder_realisations', 100))

    rho = float(parameters.get('rho', 30.0))
    kappa = float(parameters.get('kappa', 0.1))
    disorder_start = float(parameters.get('disorder_start', 0.0))
    disorder_end = float(parameters.get('disorder_end', 5.0))
    disorder_resolution = int(parameters.get('disorder_resolution', 6))
    # Number of eigenvalues to compute if we are using sparse methods
    num_eigenvalues = int(parameters.get('num_eigenvalues', 600))


    disorder_values = np.linspace(disorder_start, disorder_end, disorder_resolution)

    # Boolean flags to get eigenvalues and eigenvectors
    reteval = True
    retevec = True

    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    base_name = f"../data/3dAnderson_L{L}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realisations}_{current_date}"

    shape_4d_H = (len(disorder_values), num_disorder_realisations, L**3, L**3)
    shape_4d_SL = (len(disorder_values), num_disorder_realisations, 4*L**3, 4*L**3)
    shape_3d_H = (len(disorder_values), num_disorder_realisations, L**3)
    shape_3d_SL = (len(disorder_values), num_disorder_realisations, 4*L**3)
    shape_2d = (len(disorder_values), num_disorder_realisations)



    print(f"{cpu_count} Cores found! Using...All of them!!")
    print(f"Parameters loaded from {parameters_file}")
    print(f"Disorder from {disorder_start} to {disorder_end} with {disorder_resolution} steps")
    print(f"{num_disorder_realisations} disorder realizations per parameter set")
    print(f"Calculating {num_eigenvalues} eigenvalues per run")
    print("-"*50, flush=True)


    # NOTE - If sparse is true, then the size of these arrays needs to change
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
        modelToGetX =  ThreeDimensionalAnderson(L,0,rho,1.0)
        X = modelToGetX.X
        sparse = False
        num_eigval = num_eigenvalues
        for j, disorder in enumerate(disorder_values):
            print(f"   Disorder: {disorder}", flush=True)
            disorder_time = time.time()
            modelforkappa = ThreeDimensionalAnderson(L,disorder,rho=L//2,kappa=1.0)
            #Find largest eigenvalue to set kappa appropriately
            largest_eigenvalue = eigsh(modelforkappa.H, k=1, which='LM', return_eigenvectors=False)[0]
            kappa = largest_eigenvalue / rho
            args_list  = [(L, rho, kappa, disorder, num_eigval, X, sparse,retevec, reteval, i) for i in range(num_disorder_realisations)]
            results = list(pool.imap(single_iteration, args_list, chunksize=1))
            print(f"      Time for disorder {disorder}: {time.time() - disorder_time} seconds", flush=True)
            if retevec:
                H_eigval, spectral_localiser_eigval, H_eigvec, spectral_localiser_eigvec, seed_values = zip(*results)
                seeds[ j, :] = seed_values
                H_eigval_results[j,:, :] = np.array(H_eigval)
                spectral_localiser_eigval_results[j,:, :] = np.array(spectral_localiser_eigval)
                H_eigvec_results[j,:, :, :] = np.array(H_eigvec)
                spectral_localiser_eigvec_results[j,:, :, :] = np.array(spectral_localiser_eigvec)

                seeds.flush()
                H_eigval_results.flush()
                spectral_localiser_eigval_results.flush()
                H_eigvec_results.flush()
                spectral_localiser_eigvec_results.flush()
                continue
            else:
                H_eigval, spectral_localiser_eigval, H_IPR, spectral_localiser_IPR, seed_values = zip(*results)
                seeds[ j, :] = seed_values
                H_eigval_results[j,:, :] = np.array(H_eigval)
                spectral_localiser_eigval_results[j,:, :] = np.array(spectral_localiser_eigval)
                H_IPR_results[j,:, :] = np.array(H_IPR)
                spectral_localiser_IPR_results[j,:, :] = np.array(spectral_localiser_IPR)

                seeds.flush()
                H_eigval_results.flush()
                spectral_localiser_eigval_results.flush()
                H_IPR_results.flush()
                spectral_localiser_IPR_results.flush()

    elapsed_time = time.time() - total_time
    print(f"Total time for all calculations: {elapsed_time:.2f} seconds, or {(elapsed_time)/60:.2f} minutes, {(elapsed_time)/3600:.2f} hours", flush=True)   
    


    
    # np.savez(filename, disorder_values = disorder_values, seeds = seeds, H_eigval = H_eigval_results, spectral_localiser_eigval=spectral_localiser_eigval_results, H_IPR=H_IPR_results, spectral_localiser_IPR=spectral_localiser_IPR_results)   
    # print(f"Results saved to {filename}", flush=True)







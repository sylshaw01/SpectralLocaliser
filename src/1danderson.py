
# Makes sure nested parallelism is avoided across any underlying implementation
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import numpy as np
#mport matplotlib.pyplot as plt
from multiprocessing import cpu_count
#multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Pool
import time
import sys

cpu_count = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()

def create_localiser(L,rho,kappa,disorder):
    # Function to create the localiser matrix of a 1D Anderson model
    # Note the open boundary conditions
    # This also includes the creation of the position operator X

    # on diagonal disorder
    diag = (np.random.rand(L)-0.5) * disorder
    # off diagonal hopping terms
    off_diag = np.ones(L-1)

    H = sp.diags([off_diag,diag,off_diag],[-1,0,1],shape=(L,L),format='lil')

    #H[L-1,0] = 1
    #H[0,L-1] = 1

    row_vector = np.linspace(-rho,rho,L)
    X = sp.diags(row_vector,0,shape=(L,L),format='csr')

    localiser = sp.bmat([[-H,kappa * X],[kappa * X,H]],format='csr')

    return localiser


def find_eigenvalues(localiser, num_eigenvalues=800):
    # Function to find the positive eigenvalues of the localiser matrix
    # numpy eigsh is best for the sparse case.
    eigvals, eigvecs = eigsh(localiser, k=num_eigenvalues, sigma=0, which='LM')
    return eigvals, eigvecs



# Calculate the adjacent gap ratio r = min(s_i,s_(i+1))/max(s_i,s_(i+1))
def calculate_r(eigvals):
    # Once eigenvalues are found, calculate the r value
    eigvals_s = np.diff(eigvals)
    #min_eigvals_s = np.array([min(eigvals_s[i],eigvals_s[i+1]) for i in range(len(eigvals_s)-1)])
    #max_eigvals_s = np.array([max(eigvals_s[i],eigvals_s[i+1]) for i in range(len(eigvals_s)-1)])
    #r = min_eigvals_s / max_eigvals_s

    min_vals = np.minimum(eigvals_s[:-1],eigvals_s[1:])
    max_vals = np.maximum(eigvals_s[:-1],eigvals_s[1:])
    r = np.divide(min_vals,max_vals,out=np.zeros_like(min_vals, dtype=float),where=max_vals!=0)
    return r.mean()


def calculate_z(eigvals):    
    #eigvals = sorted(eigvals)
    #z = np.zeros(len(eigvals)-4)
    #for i in range(2,len(eigvals)-2):
    #    if abs(eigvals[i+1] - eigvals[i]) < abs(eigvals[i]-eigvals[i-1]):
    #        nn = abs(eigvals[i+1] - eigvals[i])
    #        nnn = min(min(abs(eigvals[i]-eigvals[i-1]),abs(eigvals[i+2]-eigvals[i])),abs(eigvals[i-2]-eigvals[i]))
    #    else:
    #        nn = abs(eigvals[i]-eigvals[i-1])
    #        nnn = min(min(abs(eigvals[i+1]-eigvals[i]),abs(eigvals[i-2]-eigvals[i])),abs(eigvals[i+2]-eigvals[i]))
    #    z[i-2] = nn/nnn

    eigvals = np.sort(eigvals)
    s = np.diff(eigvals)
    s_i_minus_2 = s[:-4]
    s_i_minus_1 = s[1:-3]
    s_i         = s[2:-2]
    s_i_plus_1  = s[3:-1]

    nn      = np.minimum(s_i, s_i_minus_1)
    n_other = np.maximum(s_i, s_i_minus_1)
    nnn_left  = s_i_minus_1 + s_i_minus_2
    nnn_right = s_i + s_i_plus_1
    
    nnn = np.minimum.reduce([n_other, nnn_left, nnn_right])

    z = np.divide(nn, nnn, out=np.zeros_like(nn, dtype=float), where=nnn!=0)
    return z.mean()



def single_iteration(args):
    L, rho, kappa, disorder, i = args
    localiser = create_localiser(L,rho,kappa,disorder)
    eigvals, eigvecs = find_eigenvalues(localiser, L//5)
    positive_eigvals = eigvals[eigvals > 0]
    r = calculate_r(positive_eigvals)
    z = calculate_z(positive_eigvals)
    if (i+1) % 10 == 0:
        print(f"  Iteration {i+1}/300")
    #print(f"   r value for {i+1}th iteration: {r}")
    #print(f"   z value for {i+1}th iteration: {z}")
    return r, z





print("Calculating r values for different disorder strengths and system sizes")
print("First L/5 eigenvalues are used to calculate r")

if __name__ == '__main__':

    #time the whole script

    np.random.seed(42)

    L_values = [500,1000]
    rho = 30
    kappa = 0.1
    disorder_values = np.linspace(0,5,10)
    num_iter = 500
    r_results = np.zeros((len(L_values),len(disorder_values),num_iter))
    z_results = np.zeros((len(L_values),len(disorder_values),num_iter))
    with Pool(4) as pool:
        for j, L in enumerate(L_values):

            print(f"System size L: {L}",flush = True)
            for k, disorder in enumerate(disorder_values):

                #start_time = time.time()
                #r_values_for_disorder = []
                args_list = [(L,rho,kappa,disorder,i) for i in range(num_iter)]
                #for i in range(num_iter):
                #    r = single_iteration(args_list[i])
                #    r_values_for_disorder.append(r)

                #results[j].append((disorder,np.mean(r_values_for_disorder),np.std(r_values_for_disorder)))
                results = pool.map(single_iteration,args_list, )
                r_values, z_values = zip(*results)
                r_results[j][k] = np.array(r_values)
                z_results[j][k] = np.array(z_values)
                print(f"    Disorder: {disorder}, r: {np.mean(r_values)}, z: {np.mean(z_values)}",flush = True)
                #end_time = time.time()
                #print(f"    Time taken for disorder {disorder}: {end_time - start_time} seconds")


filename = f"../data/1dAnderson_rz_results_Lmax{L_values[-1]}_iters{num_iter}.npz"
np.savez(filename, L_values=L_values, disorder_values=disorder_values, r_results=r_results, z_results=z_results)
print(f"Results saved to {filename}")

#save results to file
    # with open("1dplay_results.txt","w") as f:
    #     for i,L in enumerate(L_values):
    #         f.write(f"System size L: {L}\n")
    #         f.write("Disorder,r,stderr\n")
    #         for disorder,mean, stdev in results[i]:
    #             f.write(f"{disorder},{mean},{stdev}\n")
    #         f.write("\n")


    #plot results

    # figr, axr = plt.subplots(figsize=(8,6))
    # for i,L in enumerate(L_values):
    #     r_vals = [r_results[i][j].mean() for j in range(len(disorder_values))]
    #     r_stderr = [r_results[i][j].std()/np.sqrt(num_iter) for j in range(len(disorder_values))]
    #     axr.errorbar(disorder_values,r_vals,yerr=r_stderr,label=f"L={L}",marker='o',capthick=2)
    #     #disorders = [disorder for disorder, mean, stdev in results[i]]
    #     #r_values = [mean for disorder, mean, stdev in results[i]]
    #     #stderr_values = [stdev/np.sqrt(num_iter) for disorder, mean, stdev in results[i]]
    #     #plt.errorbar(disorders,r_values,yerr=stderr_values,label=f"L={L}",marker='o',capthick=2)

    # axr.set_xlabel("Disorder Strength")
    # axr.set_ylabel("r value")
    # axr.set_title("r value vs Disorder Strength for different System Sizes")
    # axr.legend()
    # axr.grid()
    # figr.savefig(f"1dAnderson_r_vs_disorder_Lmax{L_values[-1]}.png",dpi=300)

    # figz, axz = plt.subplots(figsize=(8,6))
    # for i,L in enumerate(L_values):
    #     z_vals = [z_results[i][j].mean() for j in range(len(disorder_values))]
    #     z_stderr = [z_results[i][j].std()/np.sqrt(num_iter) for j in range(len(disorder_values))]
    #     axz.errorbar(disorder_values,z_vals,yerr=z_stderr,label=f"L={L}",marker='o',capthick=2)

    # axz.set_xlabel("Disorder Strength")
    # axz.set_ylabel("z value")
    # axz.set_title("z value vs Disorder Strength for different System Sizes")
    # axz.legend()
    # axz.grid()
    # figz.savefig(f"1dAnderson_z_vs_disorder_Lmax{L_values[-1]}.png",dpi=300)
    
    # plt.show()



# for L in L_values:
#     print(f"System size L: {L}")
#     for disorder in disorder_values:
#         print(f"Calculating for disorder strength: {disorder}")
#         r_values_for_disorder = []
#         for i in range(num_iter):
#             print(f"  Iteration {i+1}/10")
#             localiser = create_localiser(L,rho,kappa,disorder)
#             positive_eigvals = find_eigenvalues(localiser, L/5)
#             r = calculate_r(positive_eigvals)
#             print(f"   r value for {i+1}th iteration: {r}")
#             r_values_for_disorder.append(r)
#         print(f"Disorder: {disorder}, r: {np.mean(r_values_for_disorder)}")



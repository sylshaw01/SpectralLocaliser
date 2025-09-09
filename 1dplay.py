
# Makes sure nested parallelism is avoided across any underlying implementation
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
#multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Pool

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
    eigvals, eigvecs = eigsh(localiser, k=num_eigenvalues,sigma = 0, which='LM')
    #eigvals, eigvecs = eigsh(localiser, k=num_eigenvalues, which='SM')
    positive_eigvals = eigvals[eigvals > 0]
    return positive_eigvals



# Calculate the adjacent gap ratio r = min(s_i,s_(i+1))/max(s_i,s_(i+1))
def calculate_r(eigvals):
    # Once eigenvalues are found, calculate the r value
    eigvals_s = np.diff(eigvals)
    min_eigvals_s = np.array([min(eigvals_s[i],eigvals_s[i+1]) for i in range(len(eigvals_s)-1)])
    max_eigvals_s = np.array([max(eigvals_s[i],eigvals_s[i+1]) for i in range(len(eigvals_s)-1)])
    r = min_eigvals_s / max_eigvals_s
    return r.mean()


def calculate_z(eigvals):    
    eigvals = sorted(eigvals)
    z = np.zeros(len(eigvals)-4)
    for i in range(2,len(eigvals)-2):
        if abs(eigvals[i+1] - eigvals[i]) < abs(eigvals[i]-eigvals[i-1]):
            nn = abs(eigvals[i+1] - eigvals[i])
            nnn = min(min(abs(eigvals[i]-eigvals[i-1]),abs(eigvals[i+2]-eigvals[i])),abs(eigvals[i-2]-eigvals[i]))
        else:
            nn = abs(eigvals[i]-eigvals[i-1])
            nnn = min(min(abs(eigvals[i+1]-eigvals[i]),abs(eigvals[i-2]-eigvals[i])),abs(eigvals[i+2]-eigvals[i]))
        z[i-2] = nn/nnn
    return z.mean()



def single_iteration(args):
    L, rho, kappa, disorder, i = args
    localiser = create_localiser(L,rho,kappa,disorder)
    positive_eigvals = find_eigenvalues(localiser, L//5)
    #r = calculate_r(positive_eigvals)
    z = calculate_z(positive_eigvals)
    if (i+1) % 10 == 0:
        print(f"  Iteration {i+1}/300")
    #print(f"   r value for {i+1}th iteration: {r}")
    #print(f"   z value for {i+1}th iteration: {z}")
    return z

L_values = [500,1000]
rho = 30
kappa = 0.1
disorder_values = np.linspace(0.5,5,10)
num_iter = 300



print("Calculating r values for different disorder strengths and system sizes")
print("First L/5 eigenvalues are used to calculate r")
results = [[] for _ in L_values]
if __name__ == '__main__':

    with Pool(4) as pool:
        for j, L in enumerate(L_values):
            print(f"System size L: {L}")
            for disorder in disorder_values:
                #r_values_for_disorder = []
                args_list = [(L,rho,kappa,disorder,i) for i in range(num_iter)]
                #for i in range(num_iter):
                #    r = single_iteration(args_list[i])
                #    r_values_for_disorder.append(r)

                #results[j].append((disorder,np.mean(r_values_for_disorder),np.std(r_values_for_disorder)))
                r_values_for_disorder = pool.map(single_iteration,args_list, )
                results[j].append((disorder,np.mean(r_values_for_disorder),np.std(r_values_for_disorder)))
                print(f"Disorder: {disorder}, r: {np.mean(r_values_for_disorder)}")


#save results to file
    with open("1dplay_results.txt","w") as f:
        for i,L in enumerate(L_values):
            f.write(f"System size L: {L}\n")
            f.write("Disorder,r,stderr\n")
            for disorder,mean, stdev in results[i]:
                f.write(f"{disorder},{mean},{stdev}\n")
            f.write("\n")

    #plot results
    for i,L in enumerate(L_values):
        disorders = [disorder for disorder, mean, stdev in results[i]]
        r_values = [mean for disorder, mean, stdev in results[i]]
        stderr_values = [stdev/np.sqrt(num_iter) for disorder, mean, stdev in results[i]]
        plt.errorbar(disorders,r_values,yerr=stderr_values,label=f"L={L}",marker='o',capthick=2)

    plt.xlabel("Disorder Strength")
    plt.ylabel("z value")
    plt.title("z value vs Disorder Strength for different System Sizes")
    plt.legend()
    plt.grid()
    plt.savefig("1dplay_z_vs_disorder.png",dpi=300)
    plt.show()



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



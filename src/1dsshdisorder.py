import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np
#import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import sys


cpu_count = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()


# def create_localiser(L,rho,kappa,disorder):
#     # Function to create the localiser matrix of a 1D Anderson model
#     # Note the open boundary conditions
#     # This also includes the creation of the position operator X

#     # on diagonal disorder
#     diag = (np.random.rand(L)-0.5) * disorder
#     # off diagonal hopping terms
#     off_diag = np.ones(L-1)

#     H = sp.diags([off_diag,diag,off_diag],[-1,0,1],shape=(L,L),format='lil')

#     #H[L-1,0] = 1
#     #H[0,L-1] = 1

#     row_vector = np.linspace(-rho,rho,L)
#     X = sp.diags(row_vector,0,shape=(L,L),format='csr')

#     localiser = sp.bmat([[-H,kappa * X],[kappa * X,H]],format='csr')

#     return localiser


m = 1
S = 2


def create_localiser(L,rho,kappa,disorder, X):
    mass_term = m * np.ones(L//2)
    shift_term = S * np.ones(L//2 - 1)
    A = sp.diags([mass_term,shift_term],[0,1],shape=(L//2,L//2),format='csr')
    H0 = sp.bmat([[sp.csr_matrix((L//2,L//2)),A.T],[A,sp.csr_matrix((L//2,L//2))]],format='csr')
    disorderA = (np.random.rand(L//2)-0.5) * disorder
    disorderB = (np.random.rand(L//2)-0.5) * disorder
    H_disorderA = sp.diags(disorderA,0,shape=(L//2,L//2),format='csr')
    H_disorderB = sp.diags(disorderB,0,shape=(L//2,L//2),format='csr')
    H = H0 + sp.bmat([[sp.csr_matrix((L//2,L//2)),H_disorderA],[H_disorderB,sp.csr_matrix((L//2,L//2))]],format='csr')

    
    localiser = sp.bmat([[kappa * X, H.T],[H,-kappa * X]],format='csr')

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
    #    z[i-2] = nn/nnn    eigvals = np.sort(eigvals)
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
    X, L, rho, kappa, disorder, i = args
    localiser = create_localiser(L,rho,kappa,disorder, X)
    eigvals, eigvecs = find_eigenvalues(localiser, L//5)
    positive_eigvals = eigvals[eigvals > 0]
    r = calculate_r(positive_eigvals)
    z = calculate_z(positive_eigvals)
    if i % 10 == 0:
        print(f"Completed {i} iterations")
    #print(f"   r value for {i+1}th iteration: {r}")
    #print(f"   z value for {i+1}th iteration: {z}")
    return r, z

L = 1000
rho = 30
kappa = 0.1
disorder_values = np.linspace(0,5,11)

num_iter = 100

#m_values = np.linspace(0,2,11)
m = 1.5
S=1

#localiser = create_localiser(L,rho,kappa,disorder)
#positive_eigvals = find_eigenvalues(localiser, L//5)

#print(calculate_r(positive_eigvals))
#print(calculate_z(positive_eigvals))

#plt.hist(positive_eigvals,bins=50)
#plt.title(f"Histogram of SSH Spectral Localiser eigenvalues for L={L}, disorder={disorder}")
#plt.xlabel("E")


results =[[] for _ in disorder_values]
if __name__ == '__main__':
    

    L_values = [500,1000,1500,2000]
    rho = 30
    kappa = 0.1
    disorder_values = np.linspace(0.5,5,50)
    num_iter=1000
    r_results = np.zeros((len(L_values),len(disorder_values),num_iter))
    z_results = np.zeros((len(L_values),len(disorder_values),num_iter))


    with Pool(processes=cpu_count,maxtasksperchild=10) as pool:
        for j, L in enumerate(L_values):
            print(f"System size L: {L}",flush = True)
            all_positions = np.linspace(-rho, rho, L)

            positions_A = all_positions[0::2]
            positions_B = all_positions[1::2]

            row_vector_A = sp.diags(positions_A, 0, shape=(L//2, L//2), format='csr')
            row_vector_B = sp.diags(positions_B, 0, shape=(L//2, L//2), format='csr')

            X = sp.bmat([[row_vector_A,sp.csr_matrix((L//2,L//2))],[sp.csr_matrix((L//2,L//2)),row_vector_B]],format='csr')
            for k, disorder in enumerate(disorder_values):
                print(f"    Disorder: {disorder}",flush = True)
                args_list = [(X, L,rho,kappa,disorder,i) for i in range(num_iter)]
                results = list(pool.imap(single_iteration,args_list,chunksize=1))
                r_values, z_values = zip(*results)
                r_results[j][k] = np.array(r_values)
                z_results[j][k] = np.array(z_values)
                #results[j].append((m ,np.mean(r_values_for_disorder),np.std(r_values_for_disorder)))
                print(f"    Disorder: {disorder}, r: {np.mean(r_values)}, z: {np.mean(z_values)}",flush = True)
            #for disorder in disorder_values:
                #r_values_for_disorder = []
            #    args_list = [(L,rho,kappa,disorder,i) for i in range(num_iter)]
                #for i in range(num_iter):
                #    r = single_iteration(args_list[i])
                #    r_values_for_disorder.append(r)

                #results[j].append((disorder,np.mean(r_values_for_disorder),np.std(r_values_for_disorder)))
            #    r_values_for_disorder = pool.map(single_iteration,args_list)
            #    results[j].append((disorder,np.mean(r_values_for_disorder),np.std(r_values_for_disorder)))
            #    print(f"Disorder: {disorder}, r: {np.mean(r_values_for_disorder)}")


filename = f"1dSSH_rz_results_Lmax{L_values[-1]}_iters{num_iter}.npz"
np.savez(filename, L_values=L_values, disorder_values=disorder_values, r_results=r_results, z_results=z_results)
print(f"Results saved to {filename}")

#plot r as a funbction of m

# figr, axr = plt.subplots()
# for i, L in enumerate(L_values):
#     r_means = [r_results[i][j].mean() for j in range(len(disorder_values))]
#     r_stds = [r_results[i][j].std()/np.sqrt(num_iter) for j in range(len(disorder_values))]
#     axr.errorbar(disorder_values, r_means, yerr=r_stds, marker='o', capthick=2, label=f'L={L}')

# axr.set_xlabel('Disorder Strength')
# axr.set_ylabel('r')
# axr.set_title('r vs Disorder Strength for different system sizes L')
# axr.legend()
# axr.grid()
# figr.savefig(f'ssh_r_varied_disorder_maxL{L_values[-1]}.png', dpi=300)


# figz, axz = plt.subplots()
# for i, L in enumerate(L_values):
#     z_means = [z_results[i][j].mean() for j in range(len(disorder_values))]
#     z_stds = [z_results[i][j].std()/np.sqrt(num_iter) for j in range(len(disorder_values))]
#     axz.errorbar(disorder_values, z_means, yerr=z_stds, marker='o', capthick=2, label=f'L={L}')


# axz.set_xlabel('Disorder Strength')
# axz.set_ylabel('z')
# axz.set_title('z vs Disorder Strength for different system sizes L')
# axz.legend()
# axz.grid()
# figz.savefig(f'ssh_z_varied_disorder_maxL{L_values[-1]}.png', dpi=300)
# plt.show()

# r_means = [results[j][0][1] for j in range(len(disorder_values))]
# r_stds = [results[j][0][2] for j in range(len(disorder_values))]
# plt.errorbar(disorder_values, r_means, yerr=r_stds, marker='o', capthick=2)
# plt.xlabel('Mass term m')
# plt.ylabel('z')
# plt.title(f'z vs mass term m for L={L}, disorder={disorder}')
# plt.grid()
# plt.savefig(f'ssh_z_L{L}_disorder{disorder}.png', dpi=300)
# plt.show()

#save plot
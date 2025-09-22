#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self,L,disorder, rho, kappa, X=None):
        self.L = L # system size
        self.disorder = disorder # disorder strength
        self.rho = rho # fixed value rho for position operator
        self.kappa = kappa # spectral localiser 'potential strength'
        self.X = X if X is not None else self.create_position_operator() 
        self.H = self.create_hamiltonian()
        self.SL = self.create_localiser()
    
    @abstractmethod
    def create_hamiltonian(self):
        pass

    @abstractmethod
    def create_position_operator(self):
        pass

    @abstractmethod
    def create_localiser(self):
        pass

    

    def find_eigenvalues(self, operator, num_eigenvalues=None, sparse=True):
        # finds eigenvalues and eigenvectors of a given operator. It will default to sparse methods unless told not to.
        # 
        # args:
        #  operator: the operator (typically a scipy sparse matrix) to find eigenvalues of.
        #  num_eigenvalues: the number of eigenvalues to search for. Only applies when using sparse solver.
        #  sparse: Boolean to explicitly use sparse solver or not
        #
        # returns:
        #  eigvals: array of eigenvalues
        #  eigvecs: array of eigenvectors

        if num_eigenvalues is None:
            num_eigenvalues = self.L // 5
        
        if sparse == False or num_eigenvalues == self.L:
            eigvals, eigvecs = eigh(operator.toarray())
        else:
            eigvals, eigvecs = eigsh(operator, k=num_eigenvalues, sigma=0, which='LM')

        return eigvals, eigvecs
    
    def calculate_r(self, eigvals):
        # Once eigenvalues are found, calculate the r value
        eigvals_s = np.diff(eigvals)
        min_vals = np.minimum(eigvals_s[:-1],eigvals_s[1:])
        max_vals = np.maximum(eigvals_s[:-1],eigvals_s[1:])
        r = np.divide(min_vals,max_vals,out=np.zeros_like(min_vals, dtype=float),where=max_vals!=0)
        return r.mean()
    
    def calculate_z(self, eigvals):
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
    
    def compute_statistics(self, operator, num_eigenvalues=None, sparse=True):
        eigvals, eigvecs = self.find_eigenvalues(operator,num_eigenvalues, sparse)
        positive_eigvals = eigvals[eigvals > 0]
        r = self.calculate_r(positive_eigvals)
        z = self.calculate_z(positive_eigvals)
        return r, z
    


class OneDimensionalAnderson(Model):
    
    def create_hamiltonian(self):
        L = self.L
        disorder = self.disorder

        # on diagonal disorder
        diag = (np.random.rand(L)-0.5) * disorder
        # off diagonal hopping terms
        off_diag = np.ones(L-1)

        H = sp.diags([off_diag,diag,off_diag],[-1,0,1],shape=(L,L),format='lil')

        return H
    
    def create_position_operator(self):

        row_vector = np.linspace(-self.rho,self.rho,self.L)
        X = sp.diags(row_vector,0,shape=(self.L,self.L),format='csr')
        return X

    def create_localiser(self):
        kappa = self.kappa
        X = self.X
        H = self.H

        localiser = sp.bmat([[-H, kappa * X], [kappa * X, H]], format='csr')
        #localiser = sp.csr_matrix(localiser)

        return localiser
    
class OneDimensionalSSH(Model):

    def __init__(self,L,disorder,rho,kappa,v,w,X=None):
        self.v = v # intracell hopping strength
        self.w = w # intercell hopping strength
        super().__init__(L,disorder,rho,kappa,X)
    
    @abstractmethod
    def create_hamiltonian(self):
        pass
    
    @abstractmethod
    def create_position_operator(self):
        pass
    
    def create_localiser(self):
        # creates the odd spectral localiser
        # SL(X,H,kappa) = [kappa * X, H* ]
        #                 [ H, -kappa * X]

        kappa = self.kappa
        X = self.X
        H = self.H

        localiser = sp.bmat([[kappa * X, H.T], [H, -kappa * X]], format='csr')
        
        return localiser


    def create_symmetry_reduced_localiser(self, x0=0, E0=0):
        # creates symmetry reduced spectral localiser
        # I'm not sure why we have a 'localiser' and a 'symmetry reduced localiser'
        # but it is the SRL that we use to detect topology
        #  SRL(X,H) = [kappa * (X-x0*I) - i*(H-E0*I)] @ gamma
        # where gamma is some chiral symmetry operator that anticommutes with H but commutes with X
    
        kappa = self.kappa
        X = self.X
        H = self.H
        gamma = sp.bmat([[-sp.eye(self.L//2),sp.csr_matrix((self.L//2,self.L//2))],[sp.csr_matrix((self.L//2,self.L//2)),sp.eye(self.L//2)]],format='csr')
        symmetry_reduced_localiser = ((kappa * (X-(x0*sp.eye(self.L)))) - (1j * (H-(E0*sp.eye(self.L))))) @ gamma

        return symmetry_reduced_localiser


    def calculate_winding_number(self, x0=0, E0=0):
        # calculates the winding number from the SRL
        # v(X,H) = (1/2) * sig(SRL(X,H))
        # where sig of an operator is the number of positive eigvals - number of negative eigvals

        SRL = self.create_symmetry_reduced_localiser(x0,E0)
        eigvals, eigvecs = self.find_eigenvalues(SRL, sparse=False)

        local_winding_number = (np.sum(eigvals > 0) - np.sum(eigvals < 0)) // 2
        
        return local_winding_number


class OneDimensionalSSHBlockBasis(OneDimensionalSSH):

    def create_hamiltonian(self):
        # following the basis described in notes
        # H_0 = [ 0  A* ]
        #       [ A  0  ]
        # where A = m + S 
        # Then H = H_0 + disorder
        L = self.L
        intracell_hopping = self.v * np.ones(L//2)
        intercell_hopping = self.w * np.ones(L//2 - 1)

        intracell_hopping_disordered = intracell_hopping + (np.random.rand(L//2)-0.5) * self.disorder
        intercell_hopping_disordered = intercell_hopping + (np.random.rand(L//2 - 1)-0.5) * self.disorder

        A = sp.diags([intracell_hopping_disordered,intercell_hopping_disordered],[0,1],shape=(L//2,L//2),format='csr')
        H0 = sp.bmat([[sp.csr_matrix((L//2,L//2)),A.T],[A,sp.csr_matrix((L//2,L//2))]],format='csr')
        disorderA = (np.random.rand(L//2)-0.5) * self.disorder
        disorderB = -disorderA
        H_disorderA = sp.diags(disorderA,0,shape=(L//2,L//2),format='csr')
        H_disorderB = sp.diags(disorderB,0,shape=(L//2,L//2),format='csr')

        H = H0 + sp.bmat([[sp.csr_matrix((L//2,L//2)),H_disorderA],[H_disorderB,sp.csr_matrix((L//2,L//2))]],format='csr')

        return H

    def create_position_operator(self):
        # position operator is slightly strange because all A
        # and all B sites are separated
        # as in in the position basis, a_0, a_1, a_2, ..., b_0, b_1, b_2, ...

        all_positions = np.linspace(-self.rho,self.rho,self.L)
        positions_A = all_positions[0::2]
        positions_B = all_positions[1::2]
        row_vector_A = sp.diags(positions_A, 0, shape=(self.L//2, self.L//2), format='csr')
        row_vector_B = sp.diags(positions_B, 0, shape=(self.L//2, self.L//2), format='csr')

        X = sp.bmat([[row_vector_A,sp.csr_matrix((self.L//2,self.L//2))],[sp.csr_matrix((self.L//2,self.L//2)),row_vector_B]],format='csr')

        return X
    
class OneDimensionalSSHAlternatingBasis(OneDimensionalSSH):
    # what I mean by alternating basis is
    # following the position basis that looks more like
    # a_0, b_0, a_1, b_1, a_2, b_2, ...
    # I only made the separate basis classes to double check
    # that I was implementing the block basis correctly


    def create_hamiltonian(self):

        L = self.L
        disorder = self.disorder

        intracell_hopping = self.v * np.ones(self.L//2) # intracell hopping
        intercell_hopping = self.w * np.ones(self.L//2 - 1) # intercell hopping

        intracell_hopping_disordered = intracell_hopping + (np.random.rand(self.L//2)-0.5) * disorder
        intercell_hopping_disordered = intercell_hopping + (np.random.rand(self.L//2 - 1)-0.5) * disorder


        off_diag = np.zeros(self.L-1)
        off_diag[0::2] = intracell_hopping_disordered # intracell hopping
        off_diag[1::2] = intercell_hopping_disordered # intercell hopping

        #chiral_disorder = (np.random.rand(self.L//2)-0.5) * self.disorder

        #diagonal_disorder = np.zeros(self.L)
        #diagonal_disorder[0::2] = chiral_disorder
        #diagonal_disorder[1::2] = -chiral_disorder


        H = sp.diags([off_diag,off_diag],[-1,1],shape=(self.L,self.L),format='csr')

        return H

        
    def create_position_operator(self):

        all_positions = np.linspace(-self.rho,self.rho,self.L)

        X = sp.diags(all_positions,0,shape=(self.L,self.L),format='csr')
        return X
    
    def create_symmetry_reduced_localiser(self):
        kappa = self.kappa
        X = self.X
        H = self.H

        gamma_diag = np.ones(self.L)
        gamma_diag[0::2] = -1  
        gamma = sp.diags(gamma_diag, 0, shape=(self.L, self.L), format='csr')
        
        symmetry_reduced_localiser = ((kappa * X) - (1j * H)) @ gamma

        return symmetry_reduced_localiser



class TwoDimensionalMagneticAnderson(Model):
    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        pass

    def create_localiser(self):
        pass

class ThreeDimensionalAnderson(Model):
    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        pass

    def create_localiser(self):
        pass

class ThreeDimensionalTopologicalChiralInsulator(Model):
    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        pass

    def create_localiser(self):
        pass



    


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



def single_iteration(args):
    L, rho, kappa, disorder, num_eigenvalues, X, sparse, i = args
    m = OneDimensionalAnderson(L, disorder, rho, kappa)
    hr, hz = m.compute_statistics(m.H,num_eigenvalues,sparse)
    slr, slz = m.compute_statistics(m.SL,num_eigenvalues,sparse)
    if i % 10 ==0:
        print(f"Completed {i} calculations")
    return hr, hz, slr, slz



if __name__ == "__main__":
    cpu_count = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()
    parameters_file = sys.argv[2] if len(sys.argv) > 2 else 'parameters_1dAnderson.txt'

    parameters = {}
    with open(parameters_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.split('=')
                parameters[key.strip()] = value.strip()
    L_start = int(parameters.get('L', 200))
    L_end = int(parameters.get('L_end', 400))
    L_resolution = int(parameters.get('L_step', 3))
    num_disorder_realizations = int(parameters.get('num_disorder_realizations', 100))

    rho = float(parameters.get('rho', 30.0))
    kappa = float(parameters.get('kappa', 0.1))
    disorder_start = float(parameters.get('disorder', 0.0))
    disorder_end = float(parameters.get('disorder_end', 5.0))
    disorder_resolution = int(parameters.get('disorder_step', 6))
    num_eigenvalues = float(parameters.get('num_eigenvalues', 0.2))

    L_values = np.linspace(L_start, L_end, L_resolution, dtype=int)
    disorder_values = np.linspace(disorder_start, disorder_end, disorder_resolution)
    total_calculations = len(L_values) * len(disorder_values) * num_disorder_realizations
    print(f"Total calculations to be performed: {total_calculations}")

    hr_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realizations))
    hz_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realizations))
    slr_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realizations))
    slz_results = np.zeros((len(L_values), len(disorder_values), num_disorder_realizations))

    with Pool(processes=cpu_count, maxtasksperchild=10) as pool:
        for i, L in enumerate(L_values):
            print(f"System size L: {L}", flush=True)
            modelToGetX =  OneDimensionalAnderson(L,0,rho,kappa)
            X = modelToGetX.X
            sparse = True
            num_eig = int(0.2 * L)
            if L <= 400:
                sparse = False
                num_eig = None
            for j, disorder in enumerate(disorder_values):
                print(f"   Disorder: {disorder}", flush=True)
                args_list  = [(L, rho, kappa, disorder, num_eig, X, sparse, i) for i in range(num_disorder_realizations)]
                results = list(pool.imap(single_iteration, args_list, chunksize=1))
                hr_values, hz_values, slr_values, slz_values = zip(*results)
                hr_results[i, j, :] = hr_values
                hz_results[i, j, :] = hz_values
                slr_results[i, j, :] = slr_values
                slz_results[i, j, :] = slz_values
                print(f"    Disorder {disorder} complete", flush=True)
    
    filename = f"../data/1dAnderson_L{L_start}-{L_end}_rho{rho}_kappa{kappa}_disorder{disorder_start}-{disorder_end}_numEigs{num_eigenvalues}_realizations{num_disorder_realizations}_results.npz"
    np.savez(filename, L_values = L_values, disorder_values = disorder_values, hr_results = hr_results, hz_results = hz_results, slr_results = slr_results, slz_results = slz_results)
    print(f"Results saved to {filename}", flush=True)








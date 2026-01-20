import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self,L,disorder, rho, kappa, X=None, periodic = False, energy_offset=0):
        self.L = L # system size
        self.disorder = disorder # disorder strength
        self.rho = rho # fixed value rho for position operator
        self.kappa = kappa # spectral localiser 'potential strength'
        self.X = X if X is not None else self.create_position_operator() 
        self.H = self.create_hamiltonian()
        self.spectral_localiser = self.create_localiser(energy_offset=energy_offset)
        self.H_eigval = None
        self.spectral_localiser_eigval = None
        self.H_eigvec = None
        self.spectral_localiser_eigvec = None
        self.periodic = periodic
        self.energy_offset = energy_offset
    
    @abstractmethod
    def create_hamiltonian(self):
        pass

    @abstractmethod
    def create_position_operator(self):
        pass

    @abstractmethod
    def create_localiser(self):
        pass

    

    def find_eigval(self, operator, num_eigval=None, sparse=True):
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

        if num_eigval is None:
            num_eigval = self.L // 5
        
        if sparse == False or num_eigval == self.L:
            operator_dense = operator.toarray()
            if not np.allclose(operator_dense, operator_dense.conj().T):
                eigval, eigvec = np.linalg.eig(operator_dense)
                idx = np.argsort(eigval)
                eigval = eigval[idx]
                eigvec = eigvec[:, idx]
            else:
                eigval, eigvec = eigh(operator_dense)
        else:
            try:
                k = min(num_eigval * 2, operator.shape[0] - 2)
                # Use shift-invert mode for eigenvalues near zero (band center)
                # sigma=0, which='LM' finds eigenvalues closest to zero efficiently
                # This is 10-100× faster than which='SM' for interior eigenvalues
                eigval, eigvec = eigsh(operator, k=k, sigma=0, which='LM',
                                       maxiter=10000,
                                       tol=1e-10)
            except sp.linalg.ArpackNoConvergence as e:
                print(f"ARPACK did not converge, solving densely instead")
                operator_dense = operator.toarray()
                if not np.allclose(operator_dense, operator_dense.conj().T):
                    eigval, eigvec = np.linalg.eig(operator_dense)
                    idx = np.argsort(eigval)
                    eigval = eigval[idx]
                    eigvec = eigvec[:, idx]
                else:
                    eigval, eigvec = eigh(operator_dense)
                eigval = eigval[:2*num_eigval]
                eigvec = eigvec[:, :2*num_eigval]

        if operator is self.H:
            self.H_eigval = eigval
            self.H_eigvec = eigvec
        elif operator is self.spectral_localiser:
            self.spectral_localiser_eigval = eigval
            self.spectral_localiser_eigvec = eigvec
    
    def calculate_r(self, eigval):
        # Once eigenvalues are found, calculate the r value
        eigval_s = np.diff(eigval)
        min_vals = np.minimum(eigval_s[:-1],eigval_s[1:])
        max_vals = np.maximum(eigval_s[:-1],eigval_s[1:])
        r = np.divide(min_vals,max_vals,out=np.zeros_like(min_vals, dtype=float),where=max_vals!=0)
        return r.mean()
    
    def calculate_z(self, eigval):
        s = np.diff(eigval)
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
    
    def compute_statistics(self, operator, num_eigval=None, sparse=True, tolerance=1e-7, slepc=False, return_eigval=False,return_eigvec=False):
        # given an operator, computer the r and z statistics
        #
        # args:
        #  operator: the operator (typically a scipy sparse matrix) to find eigenvalues of.
        #  num_eigenvalues: the number of eigenvalues to search for. Only applies when using sparse solver.
        #  sparse: Boolean to explicitly use sparse solver or not
        #
        # returns:
        #  r: the mean adjacent gap ratio
        #  z: the mean next nearest neighbour ratio

        # If the eigenvalues have not already been found, find them

        if operator == self.H:
            eigval = self.H_eigval
        elif operator == self.spectral_localiser:
            eigval = self.spectral_localiser_eigval
        if eigval is None:
            if slepc:
                self.find_eigval_slepc(operator,num_eigval)
            else:
                self.find_eigval(operator,num_eigval, sparse)
        sorted_eigval = np.sort(eigval)
        #print(f"Found {len(sorted_eigvals)} eigenvalues")
        positive_eigval = sorted_eigval[sorted_eigval > 0] 
        #print(f"Found {len(positive_eigvals)} positive eigenvalues")
        usable_eigval  = [positive_eigval[0]] if len(positive_eigval) > 0 else []
        
        for val in positive_eigval[1:]:
            if val - usable_eigval[-1] > tolerance:
                usable_eigval.append(val)
        #print(f"After applying tolerance of {tolerance}, count is {len(usable_eigvals)}")
        positive_eigval = np.array(usable_eigval)
        r = self.calculate_r(positive_eigval)
        z = self.calculate_z(positive_eigval)

        return r, z
        
    def compute_IPR(self, eigvec):
        # computes the inverse participation ratio for a set of eigenvectors
        # IPR(psi) = sum_i |psi_i|^4 
        ipr = np.sum(np.abs(eigvec)**4, axis=0)
        return ipr
    
    def projection_operator_lower(fermi_energy: float, operator: np.ndarray, eigval, eigvec) -> np.ndarray:

        occupied_indices = eigval <= fermi_energy
        if not np.any(occupied_indices):
            return np.zeros_like(operator)
        occupied_eigvec = eigvec[:, occupied_indices]
        projector = occupied_eigvec @ occupied_eigvec.conj().T
        return projector
    
    def calculate_local_chern_marker(self, P, X, Y, ac, ucs):
        Q = np.eye(P.shape[0]) - P

        C_mat = P @ X @ Q @ Y @ P  
        C_diag = np.diag(C_mat)
        C_diag_reshaped = C_diag.reshape(P.shape[0]//ucs, ucs)

        chern_marker_per_unit_cell = -((4 * np.pi)/ac) * np.imag(np.sum(C_diag_reshaped, axis=1))

        return chern_marker_per_unit_cell
    


class OneDimensionalAnderson(Model):
    
    def create_hamiltonian(self):
        L = self.L
        disorder = self.disorder

        # on diagonal disorder
        diag = (np.random.rand(L)-0.5) * disorder
        # off diagonal hopping terms
        off_diag = 1.0 * np.ones(L-1)

        H = sp.diags([off_diag,diag,off_diag],[-1,0,1],shape=(L,L),format='csr')

        return H
    
    def create_position_operator(self):

        row_vector = np.linspace(-self.rho,self.rho,self.L)
        X = sp.diags(row_vector,0,shape=(self.L,self.L),format='csr')
        return X

    def create_localiser(self, energy_offset=0):
        kappa = self.kappa
        X = self.X
        H = self.H

        localiser = sp.bmat([[-H, kappa * X], [kappa * X, H]], format='csr')
        #localiser = sp.csr_matrix(localiser)

        return localiser
    
class OneDimensionalSSH(Model):

    def __init__(self,L,disorder,rho,kappa,v,w,X=None, diagdisorder=0):
        self.v = v # intracell hopping strength
        self.w = w # intercell hopping strength
        self.topprop = 0
        self.diagdisorder = diagdisorder # if this is > 0 then diagonal disorder will be applied
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


    # def create_symmetry_reduced_localiser(self, x0=0, E0=0):
    #     # creates symmetry reduced spectral localiser
    #     # I'm not sure why we have a 'localiser' and a 'symmetry reduced localiser'
    #     # but it is the SRL that we use to detect topology
    #     #  SRL(X,H) = [kappa * (X-x0*I) - i*(H-E0*I)] @ gamma
    #     # where gamma is some chiral symmetry operator that anticommutes with H but commutes with X
    
    #     kappa = self.kappa
    #     X = self.X
    #     H = self.H
    #     gamma = sp.bmat([[-sp.eye(self.L//2),sp.csr_matrix((self.L//2,self.L//2))],[sp.csr_matrix((self.L//2,self.L//2)),sp.eye(self.L//2)]],format='csr')
    #     symmetry_reduced_localiser = ((kappa * (X-(x0*sp.eye(self.L)))) - (1j * (H-(E0*sp.eye(self.L))))) @ gamma

    #     return symmetry_reduced_localiser


 
    


class OneDimensionalSSHBlockBasis(OneDimensionalSSH):

    def create_hamiltonian(self):
        # following the basis described in notes
        # H_0 = [ 0  A ]
        #       [ A*  0  ]
        # where A = m + S 
        # Then H = H_0 + disorder
        # This is fixed, and works to my knowledge. The tricky part is that the spectral localiser is
        # SL = [kappa X, A,
        #      A*, -kappa X]
        L = self.L
        intracell_hopping = self.v * np.ones(L//2)
        intercell_hopping = self.w * np.ones(L//2 - 1)

        intracell_hopping_disordered = intracell_hopping + (np.random.rand(L//2)-0.5) * self.disorder
        intercell_hopping_disordered = intercell_hopping + (np.random.rand(L//2 - 1)-0.5) * self.disorder                          
        A = np.diag(intracell_hopping_disordered, k=0) + np.diag(intercell_hopping_disordered, k=-1)
        # if self.periodic:
        #     A[0,-1] = intercell_hopping_disordered[-1]
        

        
        
        H = sp.bmat([[sp.csr_matrix((L//2,L//2)),A],
                       [A.T,sp.csr_matrix((L//2,L//2))]], format='csr')
        
        return H

    def create_position_operator(self):
        # to keep block structure, this position operator is in terms of unit cells,
        # so has half the size of the hamiltonian

        # X = sp.bmat([[row_vector_A,sp.csr_matrix((self.L//2,self.L//2))],[sp.csr_matrix((self.L//2,self.L//2)),row_vector_B]],format='csr')
        X_diag = np.linspace(-self.rho,self.rho,self.L//2)
        X = sp.diags(X_diag,0, format='csr')

        return X
    
    def create_localiser(self):
        kappa = self.kappa
        X = self.X
        H = self.H

        X_block = sp.bmat([
            [kappa * X, sp.csr_matrix((self.L//2, self.L//2))],
             [sp.csr_matrix((self.L//2, self.L//2)), -kappa * X]], format='csr'
        )

        localiser = X_block + H

        return localiser
    
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

        amttopological = 0
        for i in range(1,(self.L//2)-1):
            if intercell_hopping_disordered[i] > intracell_hopping_disordered[i-1]:
                amttopological += 1
        self.topprop = amttopological/(self.L)

        off_diag = np.zeros(self.L-1)
        off_diag[0::2] = intracell_hopping_disordered # intracell hopping
        off_diag[1::2] = intercell_hopping_disordered # intercell hopping
        on_diag = (np.random.rand(self.L)-0.5) * self.diagdisorder

        #chiral_disorder = (np.random.rand(self.L//2)-0.5) * self.disorder

        #diagonal_disorder = np.zeros(self.L)
        #diagonal_disorder[0::2] = chiral_disorder
        #diagonal_disorder[1::2] = -chiral_disorder




        H = sp.diags([off_diag,on_diag, off_diag],[-1,0, 1],shape=(self.L,self.L),format='csr')

        return H
    

    def calculate_winding_number(self, x0=0, E0=0):
        # calculates the winding number from the SRL
        # v(X,H) = (1/2) * sig(SRL(X,H))
        # where sig of an operator is the number of positive eigvals - number of negative eigvals

        SRL = self.create_symmetry_reduced_localiser(x0,E0)
        eigval, eigvec = self.find_eigval(SRL, sparse=False)

        local_winding_number = (np.sum(eigval > 0) - np.sum(eigval < 0)) // 2
        
        return local_winding_number
    
    def calculate_everything(self, x0=0, E0=0, num_eigval=None, sparse=False):
        rh, zh = self.compute_statistics(self.H,num_eigval,sparse, tolerance=1e-7, slepc=False, return_eval=True, return_evec=False)
        rsl, zsl = self.compute_statistics(self.spectral_localiser,num_eigval,sparse, tolerance=1e-7, slepc=False, return_eval=True, return_evec=False)
        
        return  rh, zh,rsl, zsl

    


        
    def create_position_operator(self):

        all_positions = np.linspace(-self.rho,self.rho,self.L)

        X = sp.diags(all_positions,0,shape=(self.L,self.L),format='csr')
        return X
    
    def create_symmetry_reduced_localiser(self, x0=0, E0=0):
        kappa = self.kappa
        X = self.X
        H = self.H

        gamma_diag = np.ones(self.L)
        gamma_diag[0::2] = -1  
        gamma = sp.diags(gamma_diag, 0, shape=(self.L, self.L), format='csr')
        
        symmetry_reduced_localiser = ((kappa * X) - (1j * H)) @ gamma

        return symmetry_reduced_localiser
    

class OneDimensionalAubryAndre(Model):

    def __init__(self,L,disorder, rho, kappa, X=None, beta=(np.sqrt(5)-1)/2, theta=0):
        self.beta = beta
        self.theta = theta
    
        super().__init__(L,disorder,rho,kappa,X)

    def create_hamiltonian(self, pbc = False):
        L = self.L
        disorder = self.disorder
        beta = self.beta
        theta = self.theta

        # on diagonal disorder
        diag = disorder * np.cos((2 * np.pi *beta *  np.arange(L)) + theta)
        # off diagonal hopping terms
        off_diag = -1.0 * np.ones(L-1)

        diagonals = [off_diag, diag, off_diag]
        offsets = [-1,0,1]

        if pbc:
            diagonals.append([-1.0])
            offsets.append(L-1)
            diagonals.append([-1.0])
            offsets.append(-(L-1))

        H = sp.diags(diagonals,offsets,shape=(L,L),format='csr')

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


class TwoDimensionalMagneticAnderson(Model):

    def __init__(self,L,disorder, rho, kappa,flux=1, X=None, Y=None):
        self.L = L
        self.disorder = disorder
        self.rho = rho
        self.kappa = kappa
        self.flux = flux
        #self.X = 

    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        xvals = np.linspace(-self.rho,self.rho,self.L)
        xdiag = np.tile(xvals, self.L)
        ydiag = np.repeat(xvals,self.L)
        X = sp.spdiags(xdiag, 0, shape=(self.L**2,self.L**2), format="csr")
        Y = sp.spdiags(ydiag, 0, shape=(self.L**2,self.L**2),format = "csr")
        pass

    def create_localiser(self):
        pass



class OneDimensionalLiebModel(Model):
    def __init__(self, L, disorder, rho, kappa, X=None):
        self.L = L
        self.disorder = disorder
        self.rho = rho
        self.kappa = kappa
        self.X = X if X is not None else self.create_position_operator()
        self.H = self.create_hamiltonian()
        self.spectral_localiser = self.create_localiser()

    def create_hamiltonian(self):
        off_diag = np.ones(self.L - 1)
        diag = (np.random.rand(self.L) - 0.5) * self.disorder

        H = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(self.L, self.L), format='csr')

        return H

    def create_position_operator(self):
        row_vector = np.linspace(-self.rho,self.rho,self.L)
        X = sp.diags(row_vector,0,shape=(self.L,self.L),format='csr')
        return X 

    def create_localiser(self):
        # creates the odd spectral localiser
        # SL(X,H,kappa) = [kappa * X, H* ]
        #                 [ H, -kappa * X]

        kappa = self.kappa
        X = self.X
        H = self.H

        localiser = sp.bmat([[kappa * X, H.T], [H, -kappa * X]], format='csr')
        
        return localiser

class TwoDimensionalLiebModel(Model):
    def __init__(self, L, disorder, rho, kappa, X=None):
        self.L = L
        self.disorder = disorder
        self.rho = rho
        self.kappa = kappa
        self.X = X if X is not None else self.create_position_operator()
        self.H = self.create_hamiltonian()
        self.spectral_localiser = self.create_localiser()

    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        pass

    def create_localiser(self):
        pass

class TwoDimensionalHaldane(Model):


    def __init__(self, L, disorder, rho, kappa, X=None, t1=1.0, t2=1.0/3.0, M = 0.5, phi=np.pi/2):
        self.t1 = t1  # nearest neighbor hopping amplitude
        self.t2 = t2  # next-nearest neighbor hopping amplitude
        self.M = M # Lattice inversion-breaking potential term
        self.phi = phi  # phase for next-nearest neighbor hopping
        self.P = None  # Projection operator placeholder
        super().__init__(L,disorder,rho,kappa,X)
        

    def create_hamiltonian(self):
        """Generate the Hamiltonian matrix for the disordered Haldane model.

        Vectorized implementation that avoids Python loops over matrix indices.
        Uses numpy boolean masking to select valid hopping pairs based on
        boundary conditions.
        """
        mat_size = self.L * self.L * 2
        L2 = 2 * self.L  # Frequently used constant

        # All site indices
        all_idx = np.arange(mat_size)

        # Sublattice masks
        is_A = (all_idx % 2 == 0)  # A sublattice (even indices)
        is_B = (all_idx % 2 == 1)  # B sublattice (odd indices)

        # Position within row (0 to 2L-1) and row number (0 to L-1)
        pos_in_row = all_idx % L2
        row_num = all_idx // L2

        # ===== On-site terms (vectorized) =====
        diagonal = np.where(is_A, self.M, -self.M).astype(complex)
        if self.disorder > 0:
            diagonal += np.random.uniform(-self.disorder, self.disorder, mat_size)
        on_site_terms = sp.diags(diagonal, format='csr')

        # Collect all hopping terms: lists of (row_indices, col_indices, values)
        all_rows = []
        all_cols = []
        all_vals = []

        # ===== Nearest neighbor hopping terms =====
        # Each NN hopping has amplitude -t1

        # aLeft: A site (even) hops to B site at i-1
        # Conditions: is_A, i-1 >= 0, not at left edge of row (pos_in_row != 0)
        A_sites = all_idx[is_A]
        A_pos = pos_in_row[is_A]
        A_row = row_num[is_A]

        mask = A_pos != 0  # Not at left edge (i-1 >= 0 is implied for A sites with pos > 0)
        all_rows.append(A_sites[mask])
        all_cols.append(A_sites[mask] - 1)
        all_vals.append(np.full(np.sum(mask), -self.t1, dtype=complex))

        # bLeft: B site (odd) hops to A site at i-1
        # Conditions: is_B, i-1 >= 0 (always true for B sites since smallest B is 1)
        B_sites = all_idx[is_B]
        B_pos = pos_in_row[is_B]
        B_row = row_num[is_B]

        all_rows.append(B_sites)
        all_cols.append(B_sites - 1)
        all_vals.append(np.full(len(B_sites), -self.t1, dtype=complex))

        # aRight: A site (even) hops to B site at i+1
        # Conditions: is_A, i+1 < mat_size (always true except last A site)
        mask = A_sites + 1 < mat_size
        all_rows.append(A_sites[mask])
        all_cols.append(A_sites[mask] + 1)
        all_vals.append(np.full(np.sum(mask), -self.t1, dtype=complex))

        # bRight: B site (odd) hops to A site at i+1
        # Conditions: is_B, i+1 < mat_size, not at right edge (pos_in_row != 2L-1)
        mask = (B_sites + 1 < mat_size) & (B_pos != L2 - 1)
        all_rows.append(B_sites[mask])
        all_cols.append(B_sites[mask] + 1)
        all_vals.append(np.full(np.sum(mask), -self.t1, dtype=complex))

        # aDown: A site (even) hops to B site at i + (2L+1)
        # Conditions: is_A, i + (2L+1) < mat_size, not on last row
        offset = L2 + 1
        mask = (A_sites + offset < mat_size) & (A_row != self.L - 1)
        all_rows.append(A_sites[mask])
        all_cols.append(A_sites[mask] + offset)
        all_vals.append(np.full(np.sum(mask), -self.t1, dtype=complex))

        # bUp: B site (odd) hops to A site at i - (2L+1)
        # Conditions: is_B, i - (2L+1) >= 0, not on first row (i > 2L means row > 0)
        offset = L2 + 1
        mask = (B_sites >= offset) & (B_row > 0)
        all_rows.append(B_sites[mask])
        all_cols.append(B_sites[mask] - offset)
        all_vals.append(np.full(np.sum(mask), -self.t1, dtype=complex))

        # ===== Next-nearest neighbor hopping for A sublattice =====
        # Complex phases: exp(+i*phi) or exp(-i*phi)
        phase_plus = -self.t2 * np.exp(1j * self.phi)
        phase_minus = -self.t2 * np.exp(-1j * self.phi)

        # aNNNULeft: offset -(2L+2), phase exp(+i*phi)
        # Conditions: i - (2L+2) >= 0, pos_in_row > 0
        offset = L2 + 2
        mask = (A_sites >= offset) & (A_pos > 0)
        all_rows.append(A_sites[mask])
        all_cols.append(A_sites[mask] - offset)
        all_vals.append(np.full(np.sum(mask), phase_plus, dtype=complex))

        # aNNNLeft: offset -2, phase exp(-i*phi)
        # Conditions: i - 2 >= 0, pos_in_row != 0 (not at left edge)
        mask = (A_sites >= 2) & (A_pos != 0)
        all_rows.append(A_sites[mask])
        all_cols.append(A_sites[mask] - 2)
        all_vals.append(np.full(np.sum(mask), phase_minus, dtype=complex))

        # aNNNDRight: offset +(2L+2), phase exp(-i*phi)
        # Conditions: i + (2L+2) < mat_size, pos_in_row < 2L-2 (not at right edge)
        offset = L2 + 2
        mask = (A_sites + offset < mat_size) & (A_pos < L2 - 2)
        all_rows.append(A_sites[mask])
        all_cols.append(A_sites[mask] + offset)
        all_vals.append(np.full(np.sum(mask), phase_minus, dtype=complex))

        # aNNNDLeft: offset +2L, phase exp(+i*phi)
        # Conditions: i + 2L < mat_size
        mask = A_sites + L2 < mat_size
        all_rows.append(A_sites[mask])
        all_cols.append(A_sites[mask] + L2)
        all_vals.append(np.full(np.sum(mask), phase_plus, dtype=complex))

        # aNNNRight: offset +2, phase exp(+i*phi)
        # Conditions: i + 2 < mat_size, pos_in_row < 2L-2 (room for +2 within row)
        mask = (A_sites + 2 < mat_size) & (A_pos < L2 - 2)
        all_rows.append(A_sites[mask])
        all_cols.append(A_sites[mask] + 2)
        all_vals.append(np.full(np.sum(mask), phase_plus, dtype=complex))

        # aNNNURight: offset -2L, phase exp(-i*phi)
        # Conditions: i - 2L >= 0
        mask = A_sites >= L2
        all_rows.append(A_sites[mask])
        all_cols.append(A_sites[mask] - L2)
        all_vals.append(np.full(np.sum(mask), phase_minus, dtype=complex))

        # ===== Next-nearest neighbor hopping for B sublattice =====
        # B sublattice has OPPOSITE phases to A sublattice

        # bNNNULeft: offset -(2L+2), phase exp(-i*phi)  [opposite of A]
        # Conditions: i - (2L+2) >= 0, pos_in_row != 1 (not at left edge for B)
        offset = L2 + 2
        mask = (B_sites >= offset) & (B_pos != 1)
        all_rows.append(B_sites[mask])
        all_cols.append(B_sites[mask] - offset)
        all_vals.append(np.full(np.sum(mask), phase_minus, dtype=complex))

        # bNNNLeft: offset -2, phase exp(+i*phi)  [opposite of A]
        # Conditions: i - 2 >= 0, pos_in_row != 1 (not at left edge for B)
        mask = (B_sites >= 2) & (B_pos != 1)
        all_rows.append(B_sites[mask])
        all_cols.append(B_sites[mask] - 2)
        all_vals.append(np.full(np.sum(mask), phase_plus, dtype=complex))

        # bNNNDRight: offset +(2L+2), phase exp(+i*phi)  [opposite of A]
        # Conditions: i + (2L+2) < mat_size, not last row, not at right edge
        offset = L2 + 2
        mask = (B_sites + offset < mat_size) & (B_row != self.L - 1) & (B_pos != L2 - 1)
        all_rows.append(B_sites[mask])
        all_cols.append(B_sites[mask] + offset)
        all_vals.append(np.full(np.sum(mask), phase_plus, dtype=complex))

        # bNNNDLeft: offset +2L, phase exp(-i*phi)  [opposite of A]
        # Conditions: i + 2L < mat_size, not last row
        mask = (B_sites + L2 < mat_size) & (B_row != self.L - 1)
        all_rows.append(B_sites[mask])
        all_cols.append(B_sites[mask] + L2)
        all_vals.append(np.full(np.sum(mask), phase_minus, dtype=complex))

        # bNNNRight: offset +2, phase exp(-i*phi)  [opposite of A]
        # Conditions: i + 2 < mat_size, not at right edge
        mask = (B_sites + 2 < mat_size) & (B_pos != L2 - 1)
        all_rows.append(B_sites[mask])
        all_cols.append(B_sites[mask] + 2)
        all_vals.append(np.full(np.sum(mask), phase_minus, dtype=complex))

        # bNNNURight: offset -2L, phase exp(+i*phi)  [opposite of A]
        # Conditions: i - 2L >= 0
        mask = B_sites >= L2
        all_rows.append(B_sites[mask])
        all_cols.append(B_sites[mask] - L2)
        all_vals.append(np.full(np.sum(mask), phase_plus, dtype=complex))

        # ===== Combine all hopping terms =====
        all_rows = np.concatenate(all_rows)
        all_cols = np.concatenate(all_cols)
        all_vals = np.concatenate(all_vals)

        hopping_terms = sp.coo_matrix((all_vals, (all_rows, all_cols)),
                                       shape=(mat_size, mat_size))

        H = on_site_terms + hopping_terms

        return H  

    def create_position_operator(self):
        mat_size = self.L * self.L * 2
        i = np.arange(mat_size)

        row = i // (2 * self.L)           # Unit cell row: 0 to L-1
        unit_col = (i % (2 * self.L)) // 2  # Unit cell column: 0 to L-1
        sublattice = i % 2                 # 0 = A, 1 = B

        # Honeycomb coordinates with a1 = (1, 0), a2 = (0.5, √3/2)
        # B sites offset from A by δ = (0.5, 1/(2√3)) within unit cell
        x_diag = unit_col + 0.5 * row + 0.5 * sublattice
        y_diag = (np.sqrt(3) / 2) * row + sublattice / (2 * np.sqrt(3))
        
        # Center at (0, 0)
        x_diag -= np.mean(x_diag)
        y_diag -= np.mean(y_diag)
        
        X_operator = sp.diags(x_diag, 0, shape=(mat_size, mat_size), format='csr')
        Y_operator = sp.diags(y_diag, 0, shape=(mat_size, mat_size), format='csr')

        return [X_operator, Y_operator]

    def projection_operator_lower(self, fermi_energy, ac=3/(2*np.sqrt(3)) ):
        """Create projection operator for states below the Fermi energy.

        Uses efficient matrix multiplication P = V @ V^H instead of
        summing outer products, which eliminates O(n_occ) temporary N×N matrices.
        """
        # Get eigenvalues and eigenvectors
        if self.H_eigval is None or self.H_eigvec is None:
            self.find_eigval(self.H, sparse=False)

        # Find indices of occupied states (below Fermi energy)
        occupied_indices = np.where(self.H_eigval <= fermi_energy)[0]

        if len(occupied_indices) == 0:
            self.P = np.zeros(self.H.shape, dtype=complex)
            return

        # Create projection operator efficiently: P = V @ V^H
        # where V is the matrix of occupied eigenvectors
        occupied_eigvec = self.H_eigvec[:, occupied_indices]
        self.P = occupied_eigvec @ occupied_eigvec.conj().T

    def calculate_local_chern_marker(self, ac=3/(2*np.sqrt(3)), fermi_energy=0) -> np.ndarray:
        """Calculate local Chern marker per unit cell.

        Computes C(r) = -(4π/ac) * Im[P X (I-P) Y P]_{rr} for each unit cell.

        Memory-optimized: avoids creating explicit dense identity matrix by
        computing (I-P) @ M as M - P @ M. Also vectorizes diagonal extraction.
        """
        mat_size = self.L * self.L * 2

        # Ensure eigenvalues/eigenvectors are computed
        if self.H_eigval is None or self.H_eigvec is None:
            self.find_eigval(self.H, sparse=False)

        # Build projection operator if not already computed
        if self.P is None:
            self.projection_operator_lower(fermi_energy=fermi_energy, ac=ac)

        # Get position operator diagonals (X and Y are diagonal matrices)
        x_diag = self.X[0].diagonal()
        y_diag = self.X[1].diagonal()

        # Compute c_mat = P @ X @ (I-P) @ Y @ P step by step
        # Using (I-P) @ M = M - P @ M to avoid creating dense identity

        # Step 1: Y @ P (Y is diagonal, so this is element-wise multiply on rows)
        YP = y_diag[:, np.newaxis] * self.P  # N × N

        # Step 2: (I-P) @ YP = YP - P @ YP
        P_YP = self.P @ YP
        IminusP_YP = YP - P_YP

        # Step 3: X @ (I-P) @ YP (X is diagonal)
        X_IminusP_YP = x_diag[:, np.newaxis] * IminusP_YP  # N × N

        # Step 4: P @ X @ (I-P) @ YP
        c_mat = self.P @ X_IminusP_YP  # N × N

        # Extract diagonal and sum pairs for each unit cell (vectorized)
        c_diag = np.diag(c_mat)
        # Sum contributions from A site (even indices) and B site (odd indices)
        chern_marker_per_unit_cell = -((4*np.pi)/ac) * np.imag(c_diag[::2] + c_diag[1::2])

        return chern_marker_per_unit_cell.reshape(self.L, self.L)

    def create_localiser(self, e0=0, x0=0, y0=0):
        kappa = self.kappa
        X = self.X
        H = self.H
        # 2D spectral localiser as defined in arxiv 2411.0351
        localiser = sp.bmat([ 
            [H - e0 * sp.eye(H.shape[0]), kappa * (X[0] - x0 * sp.eye(H.shape[0])) - 1j * kappa * (X[1] - y0 * sp.eye(H.shape[0]))],
            [kappa * (X[0] - x0 * sp.eye(H.shape[0])) + 1j *kappa *  (X[1] - y0 * sp.eye(H.shape[0])), - (H - e0 * sp.eye(H.shape[0]))]
        ], format='csr')
        
        return localiser
    
    def calculate_specral_localiser_chern_marker(self):
        if self.spectral_localiser_eigval is None or self.spectral_localiser_eigvec is None:
            self.find_eigval(self.spectral_localiser, sparse=False)
        # Calculate Signatore of the spectral localiser
        sig = np.sum(self.spectral_localiser_eigval > 0) - np.sum(self.spectral_localiser_eigval < 0)
        chern_marker = sig / 2
        return chern_marker


class ThreeDimensionalLiebModel(Model):
    def __init__(self, L, disorder, rho, kappa, X=None):
        self.L = L
        self.disorder = disorder
        self.rho = rho
        self.kappa = kappa
        self.X = X if X is not None else self.create_position_operator()
        self.H = self.create_hamiltonian()
        self.spectral_localiser = self.create_localiser()

    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        pass

    def create_localiser(self):
        pass

class ThreeDimensionalAnderson(Model):


    

    def create_hamiltonian(self):
        N = self.L**3
        L = self.L

        t = -1.0

        # L = 3 example
        

        # (w, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0 ...)
        # (1, w, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, ....)
        # (0, 1, w, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, ....)

        # on diagonal disorder term
        on_diag = (np.random.rand(N) - 0.5) * self.disorder

        # Create off-diagonals for hopping terms
        # Hopping in x-direction
        # make a fully one off-diagonal
        off_diag_x = t * np.ones(N - 1)
        # then set to zero all of the elements at the far x edge
        off_diag_x[L-1::L] = 0 

        # np.arange gets [0,1,2,..,N-L-1]
        # then % (L*L) gets the position in the current yz plane
        # then // L gets the y coordinate
        # so mask_y is true for all elements at the far y edge
        y_coords = (np.arange(N-L) % (L*L)) // L
        mask_y = (y_coords == L-1)
        # Hopping in y-direction
        off_diag_y = t * np.ones(N - L)
        off_diag_y[mask_y] = 0 
        # z is easy because we don't have to worry about edges
        off_diag_z = t * np.ones(N - L*L)

        H = sp.diags(
            [
                on_diag,
                off_diag_x, off_diag_x,
                off_diag_y, off_diag_y,
                off_diag_z, off_diag_z
            ],
            [
                0,
                -1, 1,
                -L, L,
                -L*L, L*L
            ],
            shape=(N, N),
            format='csr'
        )
        
        return H

    def create_position_operator(self):
        xvals = np.linspace(-self.rho,self.rho,self.L)
        xdiag = np.tile(xvals, self.L**2) # for example if I had 3 sites with my system going between 0 and 2, my xdiag is 
                                        #               [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]
        ydiag= np.tile(np.repeat(xvals,self.L),self.L) # my ydiag is then [0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2]
        zdiag = np.repeat(xvals,self.L*self.L) # and my zdiag = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2]

        X = sp.diags(xdiag, 0, shape=(self.L**3,self.L**3),format = "csr")
        Y = sp.diags(ydiag, 0, shape=(self.L**3,self.L**3),format = "csr")
        Z = sp.diags(zdiag, 0, shape=(self.L**3,self.L**3),format = "csr")

        return [X,Y,Z]
        

    def create_localiser(self, energy_offset=0):
        pauli_x = sp.csr_matrix(np.array([[0,1],[1,0]]))
        pauli_y = sp.csr_matrix(np.array([[0,-1j],[1j,0]]))
        pauli_z = sp.csr_matrix(np.array([[1,0],[0,-1]]))
        identity_2 = sp.eye(2,format='csr')
        block1 = sp.kron(pauli_x, self.kappa * self.X[0], format='csr') + sp.kron(pauli_y, self.kappa * self.X[1], format='csr') + sp.kron(pauli_z,self.kappa * self.X[2], format='csr')
        block2 = sp.kron(identity_2, self.H, format='csr')
        localiser = sp.bmat([[-block2, block1],[block1, block2]], format='csr')
        return localiser

class ThreeDimensionalTopologicalChiralInsulator(Model):
    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        pass

    def create_localiser(self):
        pass




    

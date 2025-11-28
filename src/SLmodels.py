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
        self.eigvecs_H = None
        self.eigvecs_SL = None
        self.eigvals_H = None
        self.eigvals_SL = None
    
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
            operator_dense = operator.toarray()
            if not np.allclose(operator_dense, operator_dense.conj().T):
                eigvals, eigvecs = np.linalg.eig(operator_dense)
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
            else:
                eigvals, eigvecs = eigh(operator_dense)
        else:
            eigvals, eigvecs = eigsh(operator, k=2 * num_eigenvalues, which='SM')

        if operator is self.H:
            self.eigvals_H = eigvals
            self.eigvecs_H = eigvecs
        elif operator is self.SL:
            self.eigvals_SL = eigvals
            self.eigvecs_SL = eigvecs
    
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
    
    def compute_statistics(self, operator, num_eigenvalues=None, sparse=True, tolerance=1e-7, slepc=False, returneVals=False,returneVecs=False):
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
        if slepc:
            eigvals, eigvecs = self.find_eigvals_slepc(operator,num_eigenvalues)
        else:
            eigvals, eigvecs = self.find_eigenvalues(operator,num_eigenvalues, sparse)
        sorted_eigvals = np.sort(eigvals)
        ev = sorted_eigvals.copy()
        vectors = eigvecs.copy()
        #print(f"Found {len(sorted_eigvals)} eigenvalues")
        positive_eigvals = sorted_eigvals[sorted_eigvals > 0] 
        #print(f"Found {len(positive_eigvals)} positive eigenvalues")
        usable_eigvals  = [positive_eigvals[0]] if len(positive_eigvals) > 0 else []
        
        for val in positive_eigvals[1:]:
            if val - usable_eigvals[-1] > tolerance:
                usable_eigvals.append(val)
        #print(f"After applying tolerance of {tolerance}, count is {len(usable_eigvals)}")
        positive_eigvals = np.array(usable_eigvals)
        r = self.calculate_r(positive_eigvals)
        z = self.calculate_z(positive_eigvals)
        if returneVals==False and returneVecs==False:
            return r, z
        elif returneVals==True and returneVecs==False:
            return r, z, ev
        elif returneVecs ==True and returneVals==False:
            return r, z, vectors
        else:
            return r, z, ev, vectors
        
    def compute_IPR(self, eigvecs):
        # computes the inverse participation ratio for a set of eigenvectors
        # IPR(psi) = sum_i |psi_i|^4 
        IPRs = np.sum(np.abs(eigvecs)**4, axis=0)
        return IPRs
    
    def projection_operator_lower(fermi_energy: float, operator: np.ndarray, eigenvalues, eigenvectors) -> np.ndarray:

        occupied_indices = eigenvalues <= fermi_energy
        if not np.any(occupied_indices):
            return np.zeros_like(operator)
        occupied_eigenvectors = eigenvectors[:, occupied_indices]
        projector = occupied_eigenvectors @ occupied_eigenvectors.conj().T
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

    def create_localiser(self):
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
        # H_0 = [ 0  A* ]
        #       [ A  0  ]
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
        
        
        H = sp.bmat([[sp.csr_matrix((L//2,L//2)),A],
                       [A.T,sp.csr_matrix((L//2,L//2))]], format='csr')
        #disorderA = (np.random.rand(L//2)-0.5) * self.disorder
        #disorderB = -disorderA
        #H_disorderA = sp.diags(disorderA,0,shape=(L//2,L//2),format='csr')
        #H_disorderB = sp.diags(disorderB,0,shape=(L//2,L//2),format='csr')

        #H = H0 + sp.bmat([[sp.csr_matrix((L//2,L//2)),H_disorderA],[H_disorderB,sp.csr_matrix((L//2,L//2))]],format='csr')

        return H

    def create_position_operator(self):
        # to keep block structure, this position operator is in terms of unit cells,
        # so has half the size of the hamiltonian

        # all_positions = np.linspace(-self.rho,self.rho,self.L)
        # positions_A = all_positions[0::2]
        # positions_B = all_positions[1::2]
        # row_vector_A = sp.diags(positions_A, 0, shape=(self.L//2, self.L//2), format='csr')
        # row_vector_B = sp.diags(positions_B, 0, shape=(self.L//2, self.L//2), format='csr')

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
        eigvals, eigvecs = self.find_eigenvalues(SRL, sparse=False)

        local_winding_number = (np.sum(eigvals > 0) - np.sum(eigvals < 0)) // 2
        
        return local_winding_number
    
    def calculate_everything(self, x0=0, E0=0, num_eigenvalues=None, sparse=False):
        slevalssrl, c = self.find_eigenvalues(self.create_symmetry_reduced_localiser(0,0), num_eigenvalues, sparse)
        rh, zh, evals = self.compute_statistics(self.H,num_eigenvalues,sparse, tolerance=1e-7, slepc=False, returneVals=True, returneVecs=False)
        rsl, zsl, slevals = self.compute_statistics(self.SL,num_eigenvalues,sparse, tolerance=1e-7, slepc=False, returneVals=True, returneVecs=False)
        
        return  rh, zh,rsl, zsl,  evals, slevals, slevalssrl

    


        
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
        self.SL = self.create_localiser()

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
        self.SL = self.create_localiser()

    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        pass

    def create_localiser(self):
        pass

class ThreeDimensionalLiebModel(Model):
    def __init__(self, L, disorder, rho, kappa, X=None):
        self.L = L
        self.disorder = disorder
        self.rho = rho
        self.kappa = kappa
        self.X = X if X is not None else self.create_position_operator()
        self.H = self.create_hamiltonian()
        self.SL = self.create_localiser()

    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        pass

    def create_localiser(self):
        pass

class ThreeDimensionalAnderson(Model):


    def __init__(self, L, disorder, rho, kappa, X=None):
        # Initialize all attributes directly without calling parent __init__
        self.L = L
        self.disorder = disorder
        self.rho = rho
        self.kappa = kappa
        
        # Create position operators (list of 3) and other operators
        self.X = X if X is not None else self.create_position_operator()
        self.H = self.create_hamiltonian()
        self.SL = self.create_localiser()

    def create_hamiltonian(self):
        N = self.L**3
        L = self.L



        # L = 3 example
        

        # (w, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0 ...)
        # (1, w, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, ....)
        # (0, 1, w, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, ....)

        # on diagonal disorder term
        on_diag = (np.random.rand(N) - 0.5) * self.disorder

        # Create off-diagonals for hopping terms
        # Hopping in x-direction
        # make a fully one off-diagonal
        off_diag_x = np.ones(N - 1)
        # then set to zero all of the elements at the far x edge
        off_diag_x[L-1::L] = 0 

        # np.arange gets [0,1,2,..,N-L-1]
        # then % (L*L) gets the position in the current yz plane
        # then // L gets the y coordinate
        # so mask_y is true for all elements at the far y edge
        y_coords = (np.arange(N-L) % (L*L)) // L
        mask_y = (y_coords == L-1)
        # Hopping in y-direction
        off_diag_y = np.ones(N - L)
        off_diag_y[mask_y] = 0 
        # z is easy because we don't have to worry about edges
        off_diag_z = np.ones(N - L*L)

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
        

    def create_localiser(self):
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


class TwoDimensionalHaldane(Model):
    def create_hamiltonian(self):
        pass

    def create_position_operator(self):
        pass

    def create_localiser(self):
        pass





    

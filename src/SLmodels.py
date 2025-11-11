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
            if np.any(np.iscomplex(operator.toarray())):
                eigvals, eigvecs = np.linalg.eig(operator.toarray())
            else:
                eigvals, eigvecs = eigh(operator.toarray())
        else:
            eigvals, eigvecs = eigsh(operator, k=2 * num_eigenvalues, which='SM')

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
        evals , b = self.find_eigenvalues(self.H, num_eigenvalues, sparse)
        slevalssrl, c = self.find_eigenvalues(self.create_symmetry_reduced_localiser(0,0), num_eigenvalues, sparse)
        slevals, d = self.find_eigenvalues(self.create_localiser(), num_eigenvalues, sparse)
        rh, zh = self.compute_statistics(self.H,num_eigenvalues,sparse)
        rsl, zsl = self.compute_statistics(self.SL,num_eigenvalues,sparse)
        rsrl, zsrl = self.compute_statistics(self.create_symmetry_reduced_localiser(x0,E0),num_eigenvalues,sparse)
        
        return  rh, zh,rsl, zsl, rsrl, zsrl,  evals, slevals, slevalssrl

    


        
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

    def create_hamiltonian(self):
        L = self.L
        disorder = self.disorder

        # on diagonal disorder
        beta = (1 + 5**0.5) / 2
        diag = disorder * np.cos(2 * np.pi *beta *  np.arange(L))
        # off diagonal hopping terms
        off_diag = np.ones(L-1)

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





    

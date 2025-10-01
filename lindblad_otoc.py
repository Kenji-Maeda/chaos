import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spalin

from otoc import OTOC

def _vec(A):
    # Column-stacking (Fortran order) vectorization
    return np.asarray(A.toarray(), dtype=complex).reshape((-1, 1), order='F')

def _unvec(v, d):
    return sparse.csr_matrix(v.reshape((d, d), order='F'))

class LindbladOTOC(OTOC):
    '''
    Extract specified OTOC C^{\mu\nu}_{ij}(t) between time interval [0,T]
    '''

    def __init__(self, dissipator='boundary_dephasing_z', gamma=0.05, kappa=0.05, n_th=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dissipator = dissipator
        self.gamma = gamma
        self.kappa = kappa
        self.n_th = n_th
        self._build_heisenberg_liouvillian()

    # ---------- Jump operator factory ----------
    def _make_jumps(self):
        L = self.mfim.L
        jumps = []

        # convenience
        sz = self.mfim.sz_list
        sx = self.mfim.sx_list
        sy = self.mfim.sy_list
        # sigma^- and sigma^+ per site
        sm = [(sx[i] - 1j * sy[i]) * 0.5 for i in range(L)]
        sp = [(sx[i] + 1j * sy[i]) * 0.5 for i in range(L)]

        if self.dissipator == 'boundary_dephasing_z':
            jumps = [np.sqrt(self.gamma) * sz[0], np.sqrt(self.gamma) * sz[L-1]]
        elif self.dissipator == 'bulk_dephasing_z':
            jumps = [np.sqrt(self.gamma) * z for z in sz]
        elif self.dissipator == 'boundary_amp_damp':
            for idx in (0, L-1):
                if self.kappa > 0:
                    if (1 + self.n_th) > 0:
                        jumps.append(np.sqrt(self.kappa * (1 + self.n_th)) * sm[idx])
                    if self.n_th > 0:
                        jumps.append(np.sqrt(self.kappa * self.n_th) * sp[idx])
        elif self.dissipator == 'bulk_x_dephasing':
            jumps = [np.sqrt(self.gamma) * x for x in sx]
        else:
            raise ValueError(f"Unknown dissipator preset: {self.dissipator}")

        return jumps

    # ---------- Build Heisenberg Liouvillian L_H ----------
    def _build_heisenberg_liouvillian(self):
        H = self.mfim.H.tocsr().astype(complex)
        d = H.shape[0]
        I = sparse.eye(d, format='csr', dtype=complex)

        # Hamiltonian part: i(I ⊗ H - H^T ⊗ I)  acting on vec(A)
        L_H = 1j * (sparse.kron(I, H, 'csr') - sparse.kron(H.T, I, 'csr'))

        for Lop in self._make_jumps():
            Lop = Lop.tocsr().astype(complex)
            Ldag = Lop.getH()
            LdagL = (Ldag @ Lop).tocsr()

            # vec(L^\dagger A L) = (L^T ⊗ L^\dagger) vec(A)
            term1 = sparse.kron(Lop.T, Ldag, 'csr')
            # -1/2 {L^\dagger L, A}  ->  -1/2 [ I ⊗ (L^\dagger L) + (L^\dagger L)^T ⊗ I ]
            term2 = -0.5 * sparse.kron(I, LdagL, 'csr')
            term3 = -0.5 * sparse.kron(LdagL.T, I, 'csr')

            L_H = (L_H + term1 + term2 + term3).tocsr()

        self._L_H = L_H
        self._d = d

    # ---------- Override OTOC at time t ----------
    def otoc_t(self, t, **kwargs):
        op_dict = {
            'X': self.mfim.sx_list,
            'Y': self.mfim.sy_list,
            'Z': self.mfim.sz_list
        }
        A = op_dict[self.mu][self.i].tocsr().astype(complex)
        B = op_dict[self.nu][self.j].tocsr().astype(complex)

        # Prepare rho as in the parent class
        if self.init_state == "mixed":
            rho_pure = self.gen_pure_state([1], [[0,0,0,1]])
            self.rho = rho_pure * self.state_param + self.mfim.I / (2**self.mfim.L) * (1 - self.state_param)
        elif self.init_state == "pure":
            self.rho = self.gen_pure_state(self.state_param[0], self.state_param[1])
        elif self.init_state == "thermal":
            self.rho = self.gen_thermal_state(self.state_param, self.mfim.H)
        else:
            self.rho = self.mfim.I / (2**self.mfim.L)

        # Heisenberg evolution via superoperator
        vA0 = _vec(A)
        vAt = spalin.expm_multiply(self._L_H * float(t), vA0)  # efficient for many t's
        A_t = _unvec(vAt, self._d)

        product = (self.rho @ A_t @ B @ A_t @ B).diagonal().sum()
        return 1 - np.real(product)

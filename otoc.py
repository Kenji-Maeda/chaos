import scipy
from mfim import MFIM
import scipy.sparse.linalg as spalin
import numpy as np
import matplotlib.pyplot as plt

class OTOC:
    '''
    Extract specified OTOC C^{\mu\nu}_{ij}(t) between time interval [0,T]
    '''

    def __init__(self, **kwargs):
        
        self.mfim = kwargs.pop('mfim',0)            # mix field Ising model
        self.init_temp = kwargs.pop('init_temp', -1)    # temperature of initial thermal state (default maximally mixed)
        self.rho = kwargs.pop('rho',0)

        self.mu = kwargs.pop('mu','X')              # mu (first Pauli): 'X', 'Y', 'Z'
        self.nu = kwargs.pop('nu','X')              # nu (second Pauli): 'X', 'Y', 'Z'
        self.i = kwargs.pop('i',0)                  # i (first site): 0 to L-1
        self.j = kwargs.pop('j',0)                  # j (second site): 0 to L-1

        self.T = kwargs.pop('T',0)                  # T (end time)
        self.tstep = kwargs.pop('tstep',0)          # tstep
        self.Dt = kwargs.pop('Dt',0)                # time interval to extract chaos measure

        self.tlist = 0
        self.otoc_list = 0                          # specified OTOC between time interval [0,T]
        self.normalized_otoc_list = 0
        self.std = 0

    @classmethod
    def init(cls, **kwargs):

        L = kwargs.pop('L',4)
        J = kwargs.pop('J',0)
        hz = kwargs.pop('hz',1)
        hx = kwargs.pop('hx',1)

        mfim = MFIM.init(L=L, J=J, hz=hz, hx=hx)

        return cls(mfim=mfim, **kwargs)
    

    def analysis(self):
        '''
        Compute OTOC list, get standard deviation, extract chaos measure
        '''
        
        self.get_otoc_list()

        self.normalized_otoc_list, self.std = self.get_std()


    
    def get_otoc_list(self):

        self.tlist = np.arange(0 , self.T + self.tstep, self.tstep)
        self.otoc_list = np.zeros(len(self.tlist))

        for i, t in enumerate(self.tlist):
            self.otoc_list[i] = self.otoc_t(t)


        
    def get_std(self):
        ts = self.get_first_peak(self.otoc_list)
        time_window = self.otoc_list[ts+1:ts+int(self.Dt/(self.tlist[1]-self.tlist[0]))]
        otoc_norm = self.otoc_list/np.mean(time_window)
        std = np.sqrt(np.mean((time_window/np.mean(time_window))**2)-1)
        
        return otoc_norm, std
    
    
    def otoc_t(self, t, **kwargs):

        op_dict = {
            'X': self.mfim.sx_list,
            'Y': self.mfim.sy_list,
            'Z': self.mfim.sz_list
        }

        A = op_dict[self.mu][self.i]
        B = op_dict[self.nu][self.j]
        
        if self.init_temp == -1:
            self.rho = self.mfim.I/(2**self.mfim.L)
        else:
            self.rho = self.gen_thermal_state(self.init_temp, self.mfim.H)

        U = spalin.expm(-1j * self.mfim.H * t) # Compute U(t) = exp(-i * H * t)
        U_dag = U.getH()  # Compute U^{\dagger}
        S_i_t = U_dag @ A @ U  # Compute the time-evolved operator S_i(t)
        product = self.rho @ S_i_t @ B @ S_i_t @ B  # Compute the product S_i(t) S_j S_i(t) S_j
        otoc = 1-np.real(product.diagonal().sum())   # Compute the trace of the product:

        return otoc
    
    def plot_otoc(self):

        plt.figure(figsize=(8,4))
        plt.plot(self.tlist,self.normalized_otoc_list, label = f'L = {self.mfim.L}, hz = {self.mfim.hz}, $T_0$ = {self.init_temp}')
        plt.axhline(y=1, color='black', linestyle='--')
        plt.xlabel('$t$')
        plt.ylabel(rf"$C^{{{self.mu}{self.nu}}}_{{{self.i}{self.j}}}$")
        plt.legend()
        plt.show()
    

    @staticmethod
    def get_first_peak(lst):

        for i in range(1, len(lst) - 1):
            if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
                return i 
            
    @staticmethod
    def gen_thermal_state(temp, H, k_B=1):

        beta = 1/(k_B * temp)
        exp = spalin.expm(-beta * H)
        Z = exp.diagonal().sum()
        rho = exp/Z

        return rho

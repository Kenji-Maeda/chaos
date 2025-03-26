from otoc import OTOC
import matplotlib.pyplot as plt
import numpy as np
import itertools
import multiprocessing
import time

'''
Minimization of late-time OTOC oscillation over all pure states
'''

# Generate computational bases
bases = np.array(list(itertools.product([0, 1], repeat=4)))

def analysis()

otoc = OTOC.init()


# otoc.state_param = [x1, x2, ..., x16]
# otoc.analysis()

'''
Plan: 
initialize otoc
parallelize otoc.analysis() for various initial states
for each iteration, update otoc.init_state and otoc.state_param
'''

# otoc = OTOC.init()
# rho = otoc.gen_pure_state([1,1,1,1],[[1,1,0,1],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
# # rho = otoc.gen_pure_state([1],[[1,0,0,0]])

# print(rho.toarray())


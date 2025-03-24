from otoc import OTOC
import matplotlib.pyplot as plt
import numpy as np
import itertools

'''
Minimization of late-time OTOC oscillation over all pure states
'''


# Generate all possible combinations for [a, b, c, d] with each entry being 0 or 1.
combinations = list(itertools.product([0, 1], repeat=4))

# Convert the list of tuples into a NumPy array.
binary_array = np.array(combinations)

print(binary_array)

# otoc = OTOC.init()
# rho = otoc.gen_pure_state([1,1,1,1],[[1,1,0,1],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
# # rho = otoc.gen_pure_state([1],[[1,0,0,0]])

# print(rho.toarray())
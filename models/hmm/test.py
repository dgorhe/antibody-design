import os
import unittest
from model import HiddenMarkovModel
import pickle
from pprint import pprint
import pdb
import numpy as np
import pandas as pd

# Example data from the following notes
# https://iulg.sitehost.iu.edu/moss/hmmcalculations.pdf
obs_symbols = ['a', 'b']
state_symbols = ['s', 't']

pi = np.array([0.85, 0.15])

# Defined in section "2 Our first HMM h_1"
transition_matrix = np.array([
    [0.3, 0.7],
    [0.1, 0.9]
])

emission_matrix = np.array([
    [0.4, 0.6],
    [0.5, 0.5]
])

transition_df = pd.DataFrame(
    transition_matrix, 
    columns=state_symbols, 
    index=state_symbols
)

emission_df = pd.DataFrame(
    emission_matrix, 
    columns=obs_symbols, 
    index=state_symbols
)

# Forward probabilities, aka alpha function
abba_forward_s = [0.34, 0.06600, 0.02118, 0.00625]
abba_forward_t = [0.08, 0.15500, 0.09285, 0.04919]
abba_total_prob_forward = abba_forward_s[-1] + abba_forward_t[-1]

bab_forward_s = [0.51, 0.0644, 0.0209]
bab_forward_t = [0.08, 0.2145, 0.1190]
bab_total_prob_forward = bab_forward_s[-1] + bab_forward_t[-1]


# Backward probabilities, aka beta function
# abba_backward_s = 
# abba_backward_t = 
# abba_total_prob_backward = 

# bab_backward_s = 
# bab_backward_t = 
# bab_total_prob_backward =

def forward(starting_distribution, transition_matrix, emission_matrix, observations):
    forward_prob = []
    # pdb.set_trace()
    base_case = starting_distribution * emission_matrix.loc[:, observations[0]]
    forward_prob.append(base_case)

    for i in range(1, len(observations)):
        alpha_i = forward_prob[i - 1] * transition_matrix * emission_matrix.loc[:, observations[i]]
        forward_prob.append(alpha_i)
    
    return forward_prob

class TestHMM(unittest.TestCase):
    def baum_welch_forward(self):
        hmm = HiddenMarkovModel(encoding_path=None, transition=transition_df, emission=emission_df)
        
        expected = np.array([
            [0.34     , 0.075     ],
            [0.0657     ,0.15275   ],
            [0.020991   ,0.0917325 ],
            [0.00618822 ,0.04862648]
        ])
        observed = hmm.baum_welch_forward('abba', pi)
        norm_diff = np.linalg.norm(observed - expected, 2)
        return self.assertLessEqual(norm_diff, 1e-6)

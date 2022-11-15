import unittest
from model import HiddenMarkovModel
import pickle
import numpy as np
import pandas as pd

# Example data from the following notes
# https://iulg.sitehost.iu.edu/moss/hmmcalculations.pdf
obs_symbols = ['a', 'b']
state_symbols = ['s', 't']

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


class TestHMM(unittest.TestCase):
    # hmm = HiddenMarkovModel()
    def test_abba_forward(self, hmm):
        # calculate probability of string abba
        total_prob = 0
        return self.assertEqual(total_prob, abba_total_prob_forward)
    
    def test_bab_forward(self, hmm):
        # calculate probability of string bab
        total_prob = 0
        return self.assertEqual(total_prob, bab_total_prob_forward)
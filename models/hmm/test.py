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

class TestHMM(unittest.TestCase):
    def test_baum_welch_forward(self):
        hmm = HiddenMarkovModel(transition=transition_df, emission=emission_df, states=('s', 't'), obs_symbols=('a', 'b'))
        
        expected = np.array([
            [0.34     , 0.075     ],
            [0.0657     ,0.15275   ],
            [0.020991   ,0.0917325 ],
            [0.00618822 ,0.04862648]
        ])
        observed = hmm.baum_welch_forward('abba', pi)
        norm_diff = np.linalg.norm(observed - expected, 2)
        return self.assertLessEqual(norm_diff, 1e-6)
    
    def test_baum_welch_backward(self):
        hmm = HiddenMarkovModel(transition=transition_df, emission=emission_df, states=('s', 't'), obs_symbols=('a', 'b'))
        
        expected = np.array(
            [[0.133143, 0.127281],
            [0.2561  , 0.2487  ],
            [0.47    , 0.49    ],
            [1.      , 1.      ]
        ])
        observed = hmm.baum_welch_backward('abba')
        norm_diff = np.linalg.norm(observed - expected, 2)
        return self.assertLessEqual(norm_diff, 1e-6)

if __name__ == "__main__":
    unittest.test()
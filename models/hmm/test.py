import unittest
from model import HiddenMarkovModel
import pickle

with open("test-data.pkl", "rb") as f:
    test_data = pickle.load(f)

class TestHMM(unittest.TestCase):
    # hmm = HiddenMarkovModel()
    pass
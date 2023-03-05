from pdb import set_trace
import os
import json
import numpy as np
import pandas as pd
from typing import Iterable, Union, List
from numpy.typing import ArrayLike
from tqdm import tqdm
from utils import DIRS

class HiddenMarkovModel():
    def __init__(self, obs_symbols=None, states=None, transition=None, emission=None, ground_truth="tcr") -> None:
        self.obs_symbols = obs_symbols if obs_symbols is not None else ('0', '1', '2', '3')
        self.states = states if states is not None else self.get_state_symbols(os.path.join(DIRS["DATA"], "encoding.json"))
        self.ground_truth = ground_truth if ground_truth != "tcr" else self.get_ground_truth()

        self.transition = transition if transition is not None else self.initialize_transition(self.states)
        self.emission = emission if emission is not None else self.initialize_emission(self.obs_symbols, self.states)
        
        self.starting = None

    def get_state_symbols(self, path):
        """
        Load the amino acid symbols needed to parse antibody sequence

        Args:
            path (str): JSON filepath containing amino acid encodings

        Returns:
            tuple(str): All the one character amino acid symbols
        """
        # Load amino acid symbols
        with open(path, "r") as f:
            encoding = json.load(f)
            symbols = tuple(encoding['symbol']['one-character'].keys())

        assert len(symbols) == 21
        return symbols

    def initialize_transition(self, states) -> pd.DataFrame:
        """Create the transition matrix for the HMM model

        Args:
            states: Sequence of possible states

        Returns:
            pd.DataFrame: Transition matrix where row and column labels are states
        """
        init_array = np.random.randn(len(states), len(states))
        df = pd.DataFrame(init_array, index=states, columns=states)
        return df
    
    def initialize_emission(self, observations, states) -> pd.DataFrame:
        """Create the emission matrix for the HMM model

        Args:
            observations: Sequence of possible observations
            states: Sequence of possible states

        Returns:
            pd.DataFrame: Emission matrix where row and column labels are states
        """
        init_array = np.random.randn(len(states), len(observations))
        df = pd.DataFrame(init_array, index=states, columns=observations)
        return df

    def baum_welch_forward(
        self, 
        observations: Iterable[Union[str, int]], 
        starting: np.ndarray) -> ArrayLike:
        """
        Baum-Welch Forward Algorithm

        Args:
            observations (Iterable[Union[str, int]]): List of emitted observations
            starting (np.ndarray): Starting distribution of states (i.e. probability of starting in each state)

        Returns:
            ArrayLike: Matrix of alpha_i values of size (observations x states)
        """
        self.starting = starting
        transition_matrix = self.transition
        emission_matrix = self.emission
        
        # Set all probabilities to 0
        forward_prob = [0 for _ in range(len(observations))]
        
        # Probability of first observation is the starting distribution * emission probability
        forward_prob[0] = starting * emission_matrix.loc[:, observations[0]]

        # For observations 1, 2, ..., n the probability of the current observation
        # is the probability of transition from the previous state, to the
        # current state * the probability of emitting the current observed variable
        for i in range(1, len(observations)):
            alpha_i = forward_prob[i - 1] @ transition_matrix * emission_matrix.loc[:, observations[i]]
            forward_prob[i] = alpha_i.to_numpy()
        
        # Stack all the 1D arrays along row dimension into a matrix of size (observations x states)
        return np.vstack(forward_prob)
    
    def baum_welch_backward(self, observations):
        transition_matrix = self.transition
        emission_matrix = self.emission
        
        # Set all probabilities to 0
        backward_prob = [0 for _ in range(len(observations))]
        
        # Probability of last observation is 1 by definition
        backward_prob[-1] = np.ones(len(self.states))

        # For observations n-1, n-2, ..., 0 the probability of the current observation
        # is the probability that we emitted the subsequent observation * the probability
        # of transitioning from the current state to the subsequent state
        for i in range(len(observations) - 2, -1, -1):
            beta_i = transition_matrix @ (emission_matrix.loc[:, observations[i + 1]] * backward_prob[i + 1])
            backward_prob[i] = beta_i.to_numpy()

        # Stack all the 1D arrays along row dimension into a matrix of size (observations x states)
        return np.vstack(backward_prob)

    def baum_welch_update(self, forward_prob, backward_prob, observations):
        forward = forward_prob
        backward = backward_prob
        
        gamma = (forward * backward) / np.sum(forward * backward, axis=1)[:, np.newaxis]

        # Update transition probabilities
        xis = []
        for idx, obs in enumerate(observations[1:]):
            alpha_i = forward[idx - 1].reshape(-1, 1)
            beta_j = backward[idx].reshape(-1, 1)
            A = hmm.transition.values
            B = hmm.emission.loc[:, obs].values
            
            xi_numerator = alpha_i * A * B * beta_j
            xi_denominator = np.sum(xi_numerator, axis=None)
            
            xi = xi_numerator / xi_denominator
            xis.append(xi)
            
        a_star_numerator = np.sum(np.dstack(xis), axis=2)
        a_star_denominator = np.sum(gamma[:-1], axis=0)
        a_star = a_star_numerator / a_star_denominator
        a_star_norm = a_star / np.sum(a_star, axis=1)[:, np.newaxis]
        self.transition = pd.DataFrame(a_star_norm, index=self.states, columns=self.states)
        
        # Update emission probabilities
        b_stars = []
        for obs in set(observations):
            mask = [True if observations[i] == obs else False for i in range(len(observations))]
            numerator = np.sum(gamma[mask], axis=0)
            denominator = np.sum(gamma, axis=0)
            b_star = numerator / denominator
            b_stars.append(b_star)
        
        b_star = np.vstack(b_stars)
        b_star_norm = b_star / np.sum(b_star, axis=1)[:, np.newaxis]
        
        # TODO: Understand why b_star_norm needs to be transposed
        self.emission = pd.DataFrame(b_star_norm.T, index=self.states, columns=self.obs_symbols)
        
        # Update starting distribution
        self.starting = gamma[0]
        
    def baum_welch(self, observations_list = None, iterations=1, starting = None):
        observations_list = observations_list if observations_list is not None else self.ground_truth
        
        assert isinstance(observations_list, Iterable)
        self.starting = starting if starting is not None else np.ones(len(self.states)) / len(self.states)
        
        for observations in observations_list:
            for _ in range(iterations):
                forward = self.baum_welch_forward(observations, self.starting)
                backward = self.baum_welch_backward(observations)
                self.baum_welch_update(forward, backward, observations)

    def get_ground_truth(self) -> List[str]:
        tcr_path = os.path.join(DIRS["DATA"], "tcell_receptor_table_export_1667353882.csv")
        cols = [
            "Chain 1 Full Sequence", 
            "Chain 1 CDR1 Curated", 
            "Chain 1 CDR2 Curated", 
            "Chain 1 CDR3 Curated"
        ] 
        df_hmm = pd.read_csv(tcr_path, usecols=cols, low_memory=False).dropna()
        
        emitted = []
        for idx, row in tqdm(df_hmm.iterrows(), total=df_hmm.shape[0]):
            full, cdr1, cdr2, cdr3 = row
            cdr1_indices = list(range(full.find(cdr1), full.find(cdr1) + len(cdr1)))
            cdr2_indices = list(range(full.find(cdr2), full.find(cdr2) + len(cdr2)))
            cdr3_indices = list(range(full.find(cdr3), full.find(cdr3) + len(cdr3)))
            
            emitted_symbols = ""
            for idx, _ in enumerate(full):
                if idx in cdr1_indices:
                    emitted_symbols += "1"
                elif idx in cdr2_indices:
                    emitted_symbols += "2"
                elif idx in cdr3_indices:
                    emitted_symbols += "3"
                else:
                    emitted_symbols += "0"

            emitted.append(emitted_symbols)
        
        return emitted

if __name__ == "__main__":
    hmm = HiddenMarkovModel()
    hmm.baum_welch(['0000011100002220000333'], iterations=10, starting=None)
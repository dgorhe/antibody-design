import json
import numpy as np
import pandas as pd

class HiddenMarkovModel():
    def __init__(self, encoding_path=None, states=None, transition=None, emission=None) -> None:
        self.states = states if states is not None else ('NonCDR', 'CDR1', 'CDR2', 'CDR3')
        if encoding_path is not None:
            self.observation_symbols = self.get_observation_symbols(encoding_path)

        self.transition = transition if transition is not None else self.initialize_transition(self.states)
        self.emission = emission if emission is not None else self.initialize_emission(self.observations, self.states)

    def get_observation_symbols(self, path):
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
            observations = tuple(encoding['symbol']['one-character'].keys())

        assert len(observations) == 20
        return observations

    def initialize_transition(self, states) -> pd.DataFrame:
        """Create the transition matrix for the HMM model

        Args:
            states (Iterable[str]): Sequence of possible states (4 in our case: None, CDR1, CDR2, CDR3)

        Returns:
            pd.DataFrame: Transition matrix where row and column labels are states
        """
        init_array = np.zeros(len(states), len(states))
        df = pd.DataFrame(init_array, index=states, columns=states)
        return df

    def baum_welch_forward(self, observations, starting):
        starting_distribution = starting
        transition_matrix = self.transition
        emission_matrix = self.emission
        
        forward_prob = []
        base_case = starting_distribution * emission_matrix.loc[:, observations[0]]
        forward_prob.append(base_case)

        for i in range(1, len(observations)):
            alpha_i = forward_prob[i - 1] @ transition_matrix * emission_matrix.loc[:, observations[i]]
            forward_prob.append(alpha_i)
        
        return np.array(forward_prob)
    
    def baum_welch_backward():
        pass
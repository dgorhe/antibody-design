import json
import numpy as np
import pandas as pd
from collections import Iterable

class HiddenMarkovModel():
    def __init__(self, encoding_path, transition=None, emission=None) -> None:
        self.states = ('NonCDR', 'CDR1', 'CDR2', 'CDR3')
        self.observation_symbols = self.get_observation_symbols(encoding_path)
        self.transition = transition if transition is not None else self.initialize_transition(self.states)
        self.emission = emission if emission is not None else self.initialize_emission(self.observations, self.states)

    def get_observation_symbols(self, path: str) -> tuple(str):
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

    def initialize_transition(self, states: Iterable[str]) -> pd.DataFrame:
        """Create the transition matrix for the HMM model

        Args:
            states (Iterable[str]): Sequence of possible states (4 in our case: None, CDR1, CDR2, CDR3)

        Returns:
            pd.DataFrame: Transition matrix where row and column labels are states
        """
        init_array = np.zeros(len(states), len(states))
        df = pd.DataFrame(init_array, index=states, columns=states)
        return df

    def state_given_observation(self, obs: str) -> str:
        """
        Weighted random guess about the state from the emission matrix given emission probabilities

        Args:
            obs (str): Observation symbol (i.e. one of the amino acids)

        Returns:
            str: State symbol (None, CDR1, CDR2, CDR3)
        """
        return np.random.choice(
            self.emission.columns,
            size=None,
            replace=True,
            p=self.emission.loc[obs],
        )
    
    def observation_given_state(self, state: str) -> str:
        """
        Weighted random guess about the observation from the emission matrix given emission probabilities

        Args:
            state (str): State symbol (None, CDR1, CDR2, CDR3)

        Returns:
            str: Observation symbol (i.e. one of the amino acids)
        """
        return np.random.choice(
            list(self.emission.index),
            size=None,
            replace=True,
            p=self.emission[state],
        )

    def state_given_state(self, state: str) -> str:
        """
        Weighted random guess for state of sequence given state at time t-1

        Args:
            state (str): State symbol (None, CDR1, CDR2, CDR3)

        Returns:
            str: State symbol (None, CDR1, CDR2, CDR3)
        """
        return np.random.choice(
            list(self.transition.index),
            size=None,
            replace=True,
            p=self.transition[state],
        )

if __name__ == "__main__":
    encoding = "../../data/encoding.json"
    hmm = HiddenMarkovModel(encoding)
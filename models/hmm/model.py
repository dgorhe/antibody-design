import json
import numpy as np
import pandas as pd

class HiddenMarkovModel():
    def __init__(self, encoding_path, transition=None, emission=None) -> None:
        self.states = ('NonCDR', 'CDR1', 'CDR2', 'CDR3')
        self.observation_symbols = self.get_observation_symbols(encoding_path)
        self.transition = transition if transition is not None else self.initialize_transition(self.states)
        self.emission = emission if emission is not None else self.initialize_emission(self.observations, self.states)

    def get_observation_symbols(self, path):
        # Load amino acid symbols
        with open(path, "r") as f:
            encoding = json.load(f)
            observations = tuple(encoding['symbol']['one-character'].keys())

        assert len(observations) == 20
        return observations

    def initialize_transition(self, states):
        init_array = np.zeros(len(states), len(states))
        df = pd.DataFrame(init_array, index=states, columns=states)
        return df
    
    def initialize_transition(self, observations, states):
        init_array = np.zeros(len(observations), len(states))
        df = pd.DataFrame(init_array, index=observations, columns=states)
        return df

    def state_given_observation(self, obs):
        return np.random.choice(
            self.emission.columns,
            size=None,
            replace=True,
            p=self.emission.loc[obs],
        )
    
    def observation_given_state(self, state):
        return np.random.choice(
            list(self.emission.index),
            size=None,
            replace=True,
            p=self.emission[state],
        )

    def state_given_state(self, state):
        return np.random.choice(
            list(self.transition.index),
            size=None,
            replace=True,
            p=self.transition[state],
        )



if __name__ == "__main__":
    encoding = "../../data/encoding.json"
    hmm = HiddenMarkovModel(encoding)
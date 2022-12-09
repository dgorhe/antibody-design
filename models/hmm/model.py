import pdb
import json
import numpy as np
import pandas as pd

class HiddenMarkovModel():
    def __init__(self, obs_symbols=None, states=None, transition=None, emission=None) -> None:
        self.obs_symbols = obs_symbols

        self.transition = transition if transition is not None else self.initialize_transition(self.states)
        self.emission = emission if emission is not None else self.initialize_emission(self.observations, self.states)
        self.states = states if states is not None else ('NonCDR', 'CDR1', 'CDR2', 'CDR3')

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
            states: Sequence of possible states

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
        
        forward_prob = [0 for _ in range(len(observations))]
        forward_prob[0] = starting_distribution * emission_matrix.loc[:, observations[0]]

        for i in range(1, len(observations)):
            alpha_i = forward_prob[i - 1] @ transition_matrix * emission_matrix.loc[:, observations[i]]
            forward_prob[i] = alpha_i.to_numpy()
        
        return np.vstack(forward_prob)
    
    def baum_welch_backward(self, observations):
        transition_matrix = self.transition
        emission_matrix = self.emission
        
        backward_prob = [0 for _ in range(len(observations))]
        backward_prob[-1] = np.ones(len(self.states))

        for i in range(len(observations) - 2, -1, -1):
            beta_i = transition_matrix @ (emission_matrix.loc[:, observations[i + 1]] * backward_prob[i + 1])
            backward_prob[i] = beta_i.to_numpy()

        return np.vstack(backward_prob)

    def baum_welch_update(self, forward_prob, backward_prob, observations):
        # Getting total probability of the observation sequence for each state
        total_prob = np.sum(forward_prob[-1])
        delta = []

        for idx, obs in enumerate(observations):
            # We can only calculate until the (n-1)st observation
            if idx + 1 == len(observations):
                break
            
            # Output of forward algorithm
            alpha_curr = forward_prob[idx]
            
            # Output of backward algorithm
            beta_next = backward_prob[idx + 1]
            
            # Calculate all combinations of state transitions to current observation
            gamma_i = (alpha_curr * self.transition.T).T
            gamma_i *= self.emission.loc[:, obs] * beta_next
            gamma_i /= total_prob
            
            # Total probability of transitioning from state i to state j
            delta_i = np.sum(gamma_i, axis=1)
            delta.append(delta_i)
        
        # Final state probability is determined exclusively by the forward algorithm output
        delta.append(forward_prob[-1] / total_prob)
        
        # Turn list of 1D arrays into a single array of size (observations x states)
        return np.vstack(delta)

if __name__ == "__main__":
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
    
    hmm = HiddenMarkovModel(transition=transition_df, emission=emission_df, states=('s', 't'), obs_symbols=('a', 'b'))
    
    forward = hmm.baum_welch_forward('abba', pi)
    backward = hmm.baum_welch_backward('abba')
    delta = hmm.baum_welch_update(forward, backward, 'abba')

    print(delta)


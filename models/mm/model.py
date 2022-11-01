import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict

class MarkovModel():
    def __init__(self, filename):
        self.transition_matrix = self.generate_transition_matrix(filename)

    # Load primary sequence data
    def generate_transition_matrix(self, filename):
        data = self.load_data(filename)
        pairwise_freq, symbols = self.find_pairwise_frequencies(data)
        
        matrix = np.zeros((len(symbols), len(symbols)))
        matrix = pd.DataFrame(matrix)
        matrix.columns = [s for s in sorted(symbols)]
        matrix.index = [s for s in sorted(symbols)]

        for pair, value in tqdm(pairwise_freq.items()):
            matrix.loc[pair[0], pair[1]] = value
        
        return matrix
    
    # Load primary sequence data
    def load_data(self, filename):
        # parse file extension
        # based on file extension, parse into dataframe
        # Manipulate data as needed to create single column with whole sequence
        # Generator object from unified single column
        # Return generator object
        return filename

    # Find p_ij and p_ji for each amino acid i --> amino acid j transition
    def find_pairwise_frequencies(self, strings: list[str]):
        symbols = set()
        pairs = defaultdict(lambda: 0)

        for s in strings:
            for i in range(1, len(s)):
                # sliding window of size 2
                pair = (s[i-1], s[i])
                symbols.add(s[i-1])
                pairs[pair] += 1
        
        # Normalize by counts
        total = sum(pairs.values())
        normalized = {key: value/total for key, value in pairs.items()}

        return normalized, symbols

    def generate_sequence(self, length):
        return None

    def save_transition_matrix(self, extension='parquet'):
        if extension == 'parquet':
            self.transition_matrix.to_parquet(f'transition-matrix.{ext}')
        if extension == 'csv':
            self.transition_matrix.to_csv(f'transition-matrix.{ext}', sep=',')
        if extension == 'tsv':
            self.transition_matrix.to_tsv(f'transition-matrix.{ext}', sep='\t')


if __name__ == "__main__":
    fname = ["AABCBCB", "BCBCBCB", "AABCCCB"]
    mm = MarkovModel(fname)
    print(mm.transition_matrix)
